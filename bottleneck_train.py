# coding: utf-8

# In[]: Set GPU
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# In[]: Imports
import json
import matplotlib.pylab as plt
from glob import glob
import numpy as np
import datetime
import sys
import keras
from keras.utils import plot_model
from keras.utils import to_categorical
from PIL import Image
from sklearn.model_selection import train_test_split
from segmentation_models.backbones import get_preprocessing
from segmentation_models import Linknet, Linknet_notop, Linknet_bottleneck
from classification_models.senet import SEResNet50, preprocess_input
from keras import optimizers, callbacks
from losses import dice_coef_multiclass_loss
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

# In[]: Parameters
log = False
visualize = False

num_classes = 1
input_shape = (512, 1280, 3)
backbone = 'resnet18'

batch_size = 8
verbose = 1

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
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

if log:
    sys.stdout = Logger()

print('Date and time: {}\n'.format(loggername))

# In[]:
dataset_dir = "../../../colddata/datasets/supervisely/kamaz/KIA in summer Innopolis/"
subdirs = ["2019-04-24", "2019-05-08"]

obj_class_to_machine_color = dataset_dir + "obj_class_to_machine_color.json"

with open(obj_class_to_machine_color) as json_file:
    object_color = json.load(json_file)

ann_files = []
for subdir in subdirs:
    ann_files += [f for f in glob(dataset_dir + subdir + '/ann/' + '*.json', recursive=True)]
    
# In[]:
def get_image(path, label = False):
    img = Image.open(path)
#    img = img.resize((input_shape[1], input_shape[0]))
    img = np.array(img) 
#    img = img[:,384:896]
    if label:
        return img[...,0]
    return img  

with open(ann_files[0]) as json_file:
    data = json.load(json_file)
    tags = data['tags']
    objects = data['objects']
    
img_path = ann_files[0].replace('/ann/', '/img/').split('.json')[0]
label_path = ann_files[0].replace('/ann/', '/masks_machine/').split('.json')[0]

print("Images dtype: {}".format(get_image(img_path).dtype))
print("Labels dtype: {}\n".format(get_image(label_path, label=True).dtype))
print("Images shape: {}".format(get_image(img_path).shape))
print("Labels shape: {}\n".format(get_image(label_path, label=True).shape))

# In[]: Visualise
if visualize:
    i = 28
    x = get_image(img_path)
    y = get_image(label_path, label=True)==object_color['direct'][0]
    fig, axes = plt.subplots(nrows = 2, ncols = 1)
    axes[0].imshow(x)
    axes[1].imshow(y)
    fig.tight_layout()

# In[]: Prepare for training
test_size = 0.2

print("Train:test split = {}:{}\n".format(1-test_size, test_size))

ann_files_train, ann_files_test = train_test_split(ann_files, test_size=test_size, random_state=1)

print("Training files count: {}".format(len(ann_files_train)))
print("Testing files count: {}\n".format(len(ann_files_test)))

# In[]: 
def custom_generator(files, preprocessing_fn = None, batch_size = 1):
    
    i = 0
    
    while True:
        
        x_batch = np.zeros((batch_size, input_shape[0], input_shape[1], 3), dtype=np.uint8)
        y1_batch = np.zeros((batch_size, num_classes), dtype=np.int64)
        y2_batch = np.zeros((batch_size, input_shape[0], input_shape[1]))
        
        for b in range(batch_size):
            
            if i == len(files):
                i = 0
                
            x = get_image(ann_files[i].replace('/ann/', '/img/').split('.json')[0])
            
            with open(ann_files[i]) as json_file:
                data = json.load(json_file)
                tags = data['tags']

            y1 = 0
            if len(tags) > 0:
                for tag in range(len(tags)):
                    tag_name = tags[tag]['name']
                    if tag_name == 'offlane':
                        value = tags[tag]['value']
                        if value == '1':
                            y1 = 1
                            break
                        
            y2 = get_image(ann_files[i].replace('/ann/', '/masks_machine/').split('.json')[0], label=True)
            y2 = y2 == object_color['direct'][0]
            
            x_batch[b] = x
            y1_batch[b] = y1
            y2_batch[b] = y2
                
            i += 1
            
        x_batch = preprocessing_fn(x_batch)
        y2_batch = np.expand_dims(y2_batch, axis = -1)
        y2_batch = y2_batch.astype('int64')
    
        y_batch = {'classification_output': y1_batch, 'segmentation_output': y2_batch}
        
        yield (x_batch, y_batch)
    
# In[]:
preprocessing_fn = get_preprocessing(backbone)

train_gen = custom_generator(files = ann_files_train, 
                             preprocessing_fn = preprocessing_fn, 
                             batch_size = batch_size)

# In[]: Bottleneck
model = Linknet_bottleneck(backbone_name=backbone, input_shape=input_shape, classes=num_classes, activation='sigmoid')
#plot_model(model, to_file='linknet_bottleneck.png')
model.summary()

# In[]: 
losses = {
        "classification_output": "binary_crossentropy",
        "segmentation_output": "binary_crossentropy"
}

lossWeights = {
        "classification_output": 1.0,
        "segmentation_output": 1.0
}

optimizer = optimizers.Adam(lr = 1e-4)
model.compile(optimizer=optimizer, loss=losses, loss_weights=lossWeights, metrics=["accuracy"])

# In[]:
reduce_lr_1 = callbacks.ReduceLROnPlateau(monitor='classification_output_loss', factor = 0.5, patience = 5, verbose = 1, min_lr = 1e-8)
reduce_lr_2 = callbacks.ReduceLROnPlateau(monitor='segmentation_output_loss', factor = 0.5, patience = 5, verbose = 1, min_lr = 1e-8)

early_stopper_1 = callbacks.EarlyStopping(monitor='classification_output_loss', patience = 10, verbose = 1)
early_stopper_2 = callbacks.EarlyStopping(monitor='segmentation_output_loss', patience = 10, verbose = 1)

clbacks = [reduce_lr_1, reduce_lr_2, early_stopper_1, early_stopper_2]

if log:
    csv_logger = callbacks.CSVLogger('logs/{}.log'.format(loggername))
    model_checkpoint_1 = callbacks.ModelCheckpoint('weights/{}.hdf5'.format(loggername), monitor = 'classification_output_loss', verbose = 1, save_best_only = True, save_weights_only = True)
    model_checkpoint_2 = callbacks.ModelCheckpoint('weights/{}.hdf5'.format(loggername), monitor = 'segmentation_output_loss', verbose = 1, save_best_only = True, save_weights_only = True)
    clbacks.append(csv_logger)
    clbacks.append(model_checkpoint_1)
    clbacks.append(model_checkpoint_2)

print("Callbacks used:")
for c in clbacks:
    print("{}".format(c))

# In[]: 
steps_per_epoch = len(ann_files_train)//batch_size
epochs = 1000

history = model.fit_generator(
        generator = train_gen,
        steps_per_epoch = steps_per_epoch,
        epochs = epochs,
        verbose = verbose,
        callbacks = clbacks
        )