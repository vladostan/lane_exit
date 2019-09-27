# coding: utf-8

# In[1]:
import os
from glob import glob
import numpy as np
import datetime

# In[]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# In[]
log = True
aug_mode = 0
verbose = 2

# Get the date and time
now = datetime.datetime.now()
loggername = str(now).split(".")[0]
loggername = loggername.replace(":","-")

# Print stdout to file
import sys

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open('logs/segmentation_linknet_resnet18/{}.txt'.format(loggername), 'w')

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

#sys.stdout = open('logs/{}'.format(loggername), 'w')

print('Date and time: {}\n'.format(loggername))

# READ IMAGES AND MASKS
# In[2]:
PATH = os.path.abspath('results')

SOURCE_IMAGES = [os.path.join(PATH, "day2night_inno_cyclegan/test_latest/images")]

images = []

for si in SOURCE_IMAGES:
    images.extend(glob(os.path.join(si, "*.png")))
    
images.sort()

print("Datasets used: {}\n".format(SOURCE_IMAGES))
    
labels = []

for i in range(0, len(images), 2):
    labels.append(images[i].replace("results/day2night_inno_cyclegan/test_latest/images", "datasets/day2night_inno/labels").replace("_fake_B",""))

print(len(images))
print(len(labels))

# In[]
from PIL import Image

def get_image(path):
    img = Image.open(path)
    img = np.array(img)
    return img

def get_label(path):
    img = Image.open(path)
    img = img.resize((640,256))
    img = np.array(img)
    return img

# In[]:
class_weights = np.array([0.16626288, 1.,         1.46289384]) # for 447 day images of innopolis in 2018 

# In[]: AUGMENTATIONS
import imgaug.augmenters as iaa

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
    
def random_float(low, high):
    return np.random.random()*(high-low) + low
    
if aug_mode == 1:
    print("DOING CLASSICAL AUGMENTATION")  
elif aug_mode == 2:
    print("DOING CYCLEGAN AUGMENTATION")
elif aug_mode == 3:
    print("DOING MIXED AUGMENTATION")
elif aug_mode == 4:
    print("DOING COMBO AUGMENTATION")
else:
    print("NO AUGMENTATIONS")

if aug_mode == 1 or aug_mode == 3 or aug_mode == 4:
            
    def augment(image):
        
        mul = random_float(0.1, 0.5)
        add = np.random.randint(-100,-50)
        gamma = random_float(2,3)
        
        aug = iaa.OneOf([
                iaa.Multiply(mul = mul),
                iaa.Add(value = add),
                iaa.GammaContrast(gamma=gamma)
                ])
    
        image_augmented = aug.augment_image(image)
        
        return image_augmented

if aug_mode == 4:
    
    def augment_hard(image):
        
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

# In[]: CUSTOM GENERATORS
from keras.utils import to_categorical

num_classes = 3
input_shape = (256, 640, 3)

batch_factor = [1,2,2,3,4]

def custom_generator(images_path, labels_path, preprocessing_fn = None, aug_mode = aug_mode, batch_size = 1, validation = False):
    
    i = 0
    
    while True:
        
        if validation or aug_mode == 0:
	        x_batch = np.zeros((batch_size, input_shape[0], input_shape[1], 3))
	        y_batch = np.zeros((batch_size, input_shape[0], input_shape[1]))
        else:
            x_batch = np.zeros((batch_factor[aug_mode]*batch_size, input_shape[0], input_shape[1], 3))
            y_batch = np.zeros((batch_factor[aug_mode]*batch_size, input_shape[0], input_shape[1]))
        
        for b in range(batch_size):
            
            if i == len(labels_path):
                i = 0
                
            x = get_image(images_path[2*i+1])
            y = get_label(labels_path[i])
            
            x_batch[batch_factor[aug_mode]*b] = x
            y_batch[batch_factor[aug_mode]*b] = y

            if aug_mode == 1:
                x2 = augment(x)
                x_batch[batch_factor[aug_mode]*b+1] = x2
                y_batch[batch_factor[aug_mode]*b+1] = y
            elif aug_mode == 2:
                x2 = get_image(images_path[2*i])
                x_batch[batch_factor[aug_mode]*b+1] = x2
                y_batch[batch_factor[aug_mode]*b+1] = y 
            elif aug_mode == 3:
                x2 = augment(x)
                x3 = get_image(images_path[2*i])
                x_batch[batch_factor[aug_mode]*b+1] = x2
                x_batch[batch_factor[aug_mode]*b+2] = x3
                y_batch[batch_factor[aug_mode]*b+1] = y
                y_batch[batch_factor[aug_mode]*b+2] = y
            elif aug_mode == 4:
                x2 = augment(x)
                x3 = augment_hard(x)
                x4 = get_image(images_path[2*i])
                x_batch[batch_factor[aug_mode]*b+1] = x2
                x_batch[batch_factor[aug_mode]*b+2] = x3
                x_batch[batch_factor[aug_mode]*b+3] = x4
                y_batch[batch_factor[aug_mode]*b+1] = y
                y_batch[batch_factor[aug_mode]*b+2] = y
                y_batch[batch_factor[aug_mode]*b+3] = y
                
            i += 1
            
        x_batch = preprocessing_fn(x_batch)
        y_batch = to_categorical(y_batch, num_classes=num_classes)
        y_batch = y_batch.astype('int64')
    
        yield (x_batch, y_batch)
           
# In[ ]:
from segmentation_models.backbones import get_preprocessing

batch_size = 1

backbone = 'resnet18'

preprocessing_fn = get_preprocessing(backbone)

train_gen = custom_generator(images_path = images, 
                             labels_path = labels, 
                             preprocessing_fn = preprocessing_fn, 
                             aug_mode = aug_mode,
                             batch_size = batch_size)

# In[ ]:
# # Define model
from segmentation_models import Linknet

model = Linknet(backbone_name=backbone, input_shape=input_shape, classes=num_classes, activation='softmax')

print("Model summary:")
model.summary()

# In[ ]:
from keras import optimizers
from losses import dice_coef_multiclass_loss

learning_rate = 1e-4
optimizer = optimizers.Adam(learning_rate)

losses = [dice_coef_multiclass_loss]
metrics = ['categorical_accuracy']

print("Optimizer: {}, learning rate: {}, loss: {}, metrics: {}\n".format(optimizer, learning_rate, losses, metrics))

model.compile(optimizer = optimizer, loss = losses, metrics = metrics)

# In[]:
import tensorflow as tf
from keras import backend as K

def get_tf_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#K.set_session(get_tf_session())

# In[ ]:
from keras import callbacks

reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor = 0.5, patience = 5, verbose = 1, min_lr = 1e-8)
early_stopper = callbacks.EarlyStopping(monitor='loss', patience = 10, verbose = 1)
clbacks = [reduce_lr, early_stopper]

if log:
    csv_logger = callbacks.CSVLogger('logs/segmentation_linknet_resnet18/{}.log'.format(loggername))
    model_checkpoint = callbacks.ModelCheckpoint('weights/segmentation_linknet_resnet18/{}.hdf5'.format(loggername), monitor = 'loss', verbose = 1, save_best_only = True, save_weights_only = True)
    tensor_board = callbacks.TensorBoard(log_dir='./tblogs/segmentation_linknet_resnet18')
    clbacks.append(csv_logger)
    clbacks.append(model_checkpoint)
    clbacks.append(tensor_board)

print("Callbacks: {}\n".format(clbacks))

# In[ ]:
steps_per_epoch = len(labels)//batch_size
epochs = 1000

print("Steps per epoch: {}".format(steps_per_epoch))

print("Starting training...\n")
history = model.fit_generator(
    generator = train_gen,
    steps_per_epoch = steps_per_epoch,
    epochs = epochs,
    verbose = verbose,
    callbacks = clbacks,
    class_weight = class_weights
)
print("Finished training\n")

now = datetime.datetime.now()
loggername = str(now).split(".")[0]
loggername = loggername.replace(":","-")
print('Date and time: {}\n'.format(loggername))