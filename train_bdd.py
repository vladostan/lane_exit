# coding: utf-8

# In[]: Set GPU
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# In[]: Imports
import json
import matplotlib.pylab as plt
import numpy as np
import datetime
import sys
from keras.utils import to_categorical
from PIL import Image
from sklearn.model_selection import train_test_split
import segmentation_models as sm
from keras import optimizers, callbacks
from albumentations import (
    OneOf,
    Blur,
    RandomGamma,
    HueSaturationValue,
    RGBShift,
    RandomBrightness,
    RandomContrast,
    MedianBlur,
    CLAHE
)

import tensorflow as tf
from keras import backend as k
 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
k.tensorflow_backend.set_session(tf.Session(config=config))

# In[]: Parameters
log = True
visualize = False
aug = False

classification_classes = 4
segmentation_classes = 3

input_shape = (320, 640, 3)

backbone = 'resnet50'

random_state = 28

batch_factor = 1
batch_size_init = 12
batch_size = batch_size_init//batch_factor

verbose = 2

# In[]: Logger
now = datetime.datetime.now()
loggername = str(now).split(".")[0]
loggername = loggername.replace(":","-")

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open('logs/{}.txt'.format(loggername), 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass    

if log:
    sys.stdout = Logger()

print('Date and time: {}\n'.format(loggername))
print("LOG: {}\nAUG: {}\nCLASSIFICATION CLASSES: {}\nSEGMENTATION CLASSES: {}\nINPUT SHAPE: {}\nBACKBONE: {}\nRANDOM STATE: {}\nBATCH SIZE: {}\n".format(log, aug, classification_classes, segmentation_classes, input_shape, backbone, random_state, batch_size))

# In[]:
dataset_dir = "../../datasets/bdd/"

#ann_file_train_val = dataset_dir + "labels/" + 'bdd100k_labels_images_train_part.json'
ann_file_train_val = dataset_dir + "labels/" + 'bdd100k_labels_images_train.json'
#ann_file_test = dataset_dir + "labels/" + 'bdd100k_labels_images_val.json'    

# In[]:
def get_image(path):
    img = Image.open(path)
    img = img.resize((640,360))
    img = np.array(img)
    return img[20:-20]

with open(ann_file_train_val) as json_file:
    data_train_val = json.load(json_file)
    
#with open(ann_file_test) as json_file:
#    data_test = json.load(json_file)
    
# In[]:
i = 75

data = data_train_val[i]

attributes = data['attributes']
scene = attributes['scene']
timeofday = attributes['timeofday']
weather = attributes['weather']

labels = data['labels']
for label in labels:
    category = label['category']
    attributes = label['attributes']
    if category == 'drivable area':
        if attributes['areaType'] == 'direct':
            print("direct")
        elif attributes['areaType'] == 'alternative':
            print("alternative")
    
name = data['name']

img_path = dataset_dir + 'images/train/' + name
label_path = dataset_dir + 'drivable_maps/labels/train/' + name.split('.jpg')[0] + "_drivable_id.png"

print("Images dtype: {}".format(get_image(img_path).dtype))
print("Labels dtype: {}\n".format(get_image(label_path).dtype))
print("Images shape: {}".format(get_image(img_path).shape))
print("Labels shape: {}\n".format(get_image(label_path).shape))

# In[]: Visualise
if visualize:
    x = get_image(img_path)
    y = get_image(label_path)
    fig, axes = plt.subplots(nrows = 2, ncols = 1)
    axes[0].imshow(x)
    axes[1].imshow(y)
    fig.tight_layout()

# In[]: Prepare for training
val_size = 0.10

print("Train:Val split = {}:{}\n".format(1-val_size, val_size))

data_train, data_val = train_test_split(data_train_val, test_size=val_size, random_state=random_state)
del(data_train_val)

print("Training files count: {}".format(len(data_train)))
print("Validation files count: {}".format(len(data_val)))
#print("Test files count: {}".format(len(data_test)))

# In[]:
def augment(image):
    
    aug = OneOf([
        Blur(blur_limit=5, p=1.),
        RandomGamma(gamma_limit=(50, 150), p=1.),
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.),
        RGBShift(r_shift_limit=15, g_shift_limit=5, b_shift_limit=15, p=1.),
        RandomBrightness(limit=.25, p=1.),
        RandomContrast(limit=.25, p=1.),
        MedianBlur(blur_limit=5, p=1.),
        CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.)
        ], p=1.)

    augmented = aug(image=image)
    image_augmented = augmented['image']
    
    return image_augmented
    
# In[]: 
def train_generator(files, preprocessing_fn = None, aug = False, batch_size = 1):
    
    i = 0
    
    while True:
        
        x_batch = np.zeros((batch_factor*batch_size, input_shape[0], input_shape[1], 3), dtype=np.uint8)
        y1_batch = np.zeros((batch_factor*batch_size, classification_classes), dtype=np.int64)
        y2_batch = np.zeros((batch_factor*batch_size, input_shape[0], input_shape[1]))
        
        for b in range(batch_size):
            
            if i == len(files):
                i = 0
                
            data = files[i]
            timeofday = data['attributes']['timeofday']              
            name = data['name']
                                
            x = get_image(dataset_dir + 'images/train/' + name)

            if timeofday == "undefined":
                y1 = 0
            elif timeofday == "daytime":
                y1 = 1
            elif timeofday == "dawn/dusk":
                y1 = 2
            elif timeofday == "night":
                y1 = 3
            else:
                raise ValueError("Impossible value for time of day class")

            y1 = to_categorical(y1, num_classes=classification_classes)

            y2 = get_image(dataset_dir + 'drivable_maps/labels/train/' + name.split('.jpg')[0] + "_drivable_id.png")
            
            x_batch[batch_factor*b] = x
            y1_batch[batch_factor*b] = y1
            y2_batch[batch_factor*b] = y2
            
            if aug == 1:
                x2 = augment(x)
                x_batch[batch_factor*b+1] = x2
                y1_batch[batch_factor*b+1] = y1
                y2_batch[batch_factor*b+1] = y2
                
            i += 1
            
        x_batch = preprocessing_fn(x_batch)
        y2_batch = to_categorical(y2_batch, num_classes=segmentation_classes)
        y2_batch = y2_batch.astype('int64')
    
        y_batch = {'classification_output': y1_batch, 'segmentation_output': y2_batch}
        
        yield (x_batch, y_batch)
        
def val_generator(files, preprocessing_fn = None, batch_size = 1):
    
    i = 0
    
    while True:
        
        x_batch = np.zeros((batch_size, input_shape[0], input_shape[1], 3), dtype=np.uint8)
        y1_batch = np.zeros((batch_size, classification_classes), dtype=np.int64)
        y2_batch = np.zeros((batch_size, input_shape[0], input_shape[1]))
        
        for b in range(batch_size):
            
            if i == len(files):
                i = 0
                
            data = files[i]
            timeofday = data['attributes']['timeofday']              
            name = data['name']
                                
            x = get_image(dataset_dir + 'images/train/' + name)

            if timeofday == "undefined":
                y1 = 0
            elif timeofday == "daytime":
                y1 = 1
            elif timeofday == "dawn/dusk":
                y1 = 2
            elif timeofday == "night":
                y1 = 3
            else:
                raise ValueError("Impossible value for time of day class")
            
            y1 = to_categorical(y1, num_classes=classification_classes)

            y2 = get_image(dataset_dir + 'drivable_maps/labels/train/' + name.split('.jpg')[0] + "_drivable_id.png")
            
            x_batch[b] = x
            y1_batch[b] = y1
            y2_batch[b] = y2
                
            i += 1
            
        x_batch = preprocessing_fn(x_batch)
        y2_batch = to_categorical(y2_batch, num_classes=segmentation_classes)
        y2_batch = y2_batch.astype('int64')
    
        y_batch = {'classification_output': y1_batch, 'segmentation_output': y2_batch}
        
        yield (x_batch, y_batch)
    
# In[]:
preprocessing_fn = sm.get_preprocessing(backbone)

train_gen = train_generator(files = data_train, 
                             preprocessing_fn = preprocessing_fn, 
                             aug = aug,
                             batch_size = batch_size)

if val_size > 0:
    val_gen = val_generator(files = data_val, 
                             preprocessing_fn = preprocessing_fn, 
                             batch_size = batch_size_init)

# In[]: Bottleneck
model = sm.Linknet_bottleneck_crop(backbone_name=backbone, input_shape=input_shape, classification_classes=classification_classes, segmentation_classes = segmentation_classes, classification_activation = 'softmax', segmentation_activation='softmax')
model.summary()

# In[]: 
losses = {
        "classification_output": "categorical_crossentropy",
        "segmentation_output": sm.losses.dice_loss

}

loss_weights = {
        "classification_output": 1.0,
        "segmentation_output": 1.0
}

optimizer = optimizers.Adam(lr = 1e-4)
model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=["accuracy"])

# In[]:
monitor = 'val_' if val_size > 0 else ''

reduce_lr = callbacks.ReduceLROnPlateau(monitor = monitor+'loss', factor = 0.5, patience = 5, verbose = 1, min_lr = 1e-8)
early_stopper = callbacks.EarlyStopping(monitor = monitor+'loss', patience = 10, verbose = 1)

clbacks = [reduce_lr, early_stopper]

if log:
    csv_logger = callbacks.CSVLogger('logs/{}.log'.format(loggername))
    model_checkpoint = callbacks.ModelCheckpoint('weights/{}.hdf5'.format(loggername), monitor = monitor+'loss', verbose = 1, save_best_only = True, save_weights_only = True)
    if not os.path.exists:
        os.mkdir('tflogs/{}'.format(loggername))
    tensorboard = callbacks.tensorboard_v1.TensorBoard(log_dir='tflogs/{}'.format(loggername))
    
    clbacks.append(csv_logger)
    clbacks.append(model_checkpoint)
    clbacks.append(tensorboard)


print("Callbacks used:")
for c in clbacks:
    print("{}".format(c))

# In[]: 
steps_per_epoch = len(data_train)//batch_size
validation_steps = len(data_val)//batch_size
epochs = 1000

print("Steps per epoch: {}".format(steps_per_epoch))

print("Starting training...\n")
if val_size > 0:
    history = model.fit_generator(
            generator = train_gen,
            steps_per_epoch = steps_per_epoch,
            epochs = epochs,
            verbose = verbose,
            callbacks = clbacks,
            validation_data = val_gen,
            validation_steps = validation_steps
    )
else:
    history = model.fit_generator(
            generator = train_gen,
            steps_per_epoch = steps_per_epoch,
            epochs = epochs,
            verbose = verbose,
            callbacks = clbacks
    )
print("Finished training\n")

now = datetime.datetime.now()
loggername = str(now).split(".")[0]
loggername = loggername.replace(":","-")
print('Date and time: {}\n'.format(loggername))