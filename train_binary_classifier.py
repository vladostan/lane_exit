# -*- coding: utf-8 -*-

# In[]:
import keras
from classification_models.senet import SEResNet50, preprocess_input
import matplotlib.pylab as plt
from glob import glob
import numpy as np
import datetime
import os
import sys
from keras.utils import to_categorical
from PIL import Image
from sklearn.model_selection import train_test_split
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

# In[]:
log = False
verbose = 1
batch_size = 1
num_classes = 2
input_shape = (224, 224, 3)

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
images_path = "data/classification/binary/"
images = [y for x in os.walk(images_path) for y in glob(os.path.join(x[0], '*.jpg'))]

# In[]:
def get_image(path):
    img = Image.open(path)
    img = img.resize((input_shape[1], input_shape[0]))
    img = np.array(img) 
    return img 

# In[]:
test_size = 0.2

print("Train:test split = {}:{}\n".format(1-test_size, test_size))

images_train, images_test = train_test_split(images, test_size=test_size, random_state=28)

print("Training images count: {}".format(len(images_train)))
print("Testing images count: {}".format(len(images_test)))

# In[]: Class weighting
temp = 0
for im in images_train:
    if 'temp' in im:
        temp += 1

cw = np.array([temp, len(images_train) - temp])
cw = np.median(cw/sum(cw))/(cw/sum(cw))
class_weights = {0: cw[0], 1: cw[1]}

# In[]: Augmentor
aug = OneOf([
        Blur(blur_limit=5, p=1.),
        RandomGamma(gamma_limit=(50, 150), p=1.),
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.),
        RGBShift(r_shift_limit=15, g_shift_limit=5, b_shift_limit=15, p=1.),
        RandomBrightness(limit=.25, p=1.),
        RandomContrast(limit=.25, p=1.),
        MedianBlur(blur_limit=5, p=1.),
        CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.)
        ], p=.5)

def augment(image, aug=aug):
    augmented = aug(image=image)
    image_augmented = augmented['image']
    return image_augmented

# In[]: Custom generator
def custom_generator(images_path, batch_size = 1, validate = False):
    
    i = 0
    
    while True:
        
        x_batch = np.zeros((batch_size, input_shape[0], input_shape[1], 3))
        y_batch = np.zeros((batch_size, num_classes), dtype=np.int64)
        
        for b in range(batch_size):
            
            if i == len(images_path):
                i = 0
                
            x = get_image(images_path[i])
            y = 1 if 'temp' in images_path[i] else 0
            y = to_categorical(y, num_classes=num_classes)
            
            if not validate:
                x = augment(x)
            
            x_batch[b] = x
            y_batch[b] = y
                
            i += 1
            
        x_batch = preprocess_input(x_batch)
    
        yield (x_batch, y_batch)

# In[ ]:
train_gen = custom_generator(images_path = images_train, 
                             batch_size = batch_size)

val_gen = custom_generator(images_path = images_test,
                         batch_size = batch_size,
                         validate = True)

# In[]:
base_model = SEResNet50(input_shape=input_shape, weights='imagenet', classes=num_classes, include_top=False)
x = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(num_classes, activation='softmax')(x)
model = keras.models.Model(inputs=[base_model.input], outputs=[output])

model.summary()

learning_rate = 1e-3
optimizer = optimizers.Adam(lr = learning_rate)

losses = ['binary_crossentropy']
metrics = ['accuracy']

print("Optimizer: {}, learning rate: {}, loss: {}, metrics: {}\n".format(optimizer, learning_rate, losses, metrics))

model.compile(optimizer = optimizer, loss = losses, metrics = metrics)

# In[]:
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor = 0.5, patience = 4, verbose = 1, min_lr = 1e-8)
early_stopper = callbacks.EarlyStopping(monitor='val_loss', patience = 8, verbose = 1)
clbacks = [reduce_lr, early_stopper]

if log:
    csv_logger = callbacks.CSVLogger('logs/{}.log'.format(loggername))
    model_checkpoint = callbacks.ModelCheckpoint('weights/{}.hdf5'.format(loggername), monitor = 'val_loss', verbose = 1, save_best_only = True, save_weights_only = True)
    tensor_board = callbacks.TensorBoard(log_dir='./tblogs')
    clbacks.append(csv_logger)
    clbacks.append(model_checkpoint)
    clbacks.append(tensor_board)

print("Callbacks: {}\n".format(clbacks))

# In[]:
steps_per_epoch = len(images_train)//batch_size
validation_steps = len(images_test)//batch_size
epochs = 1000

print("Steps per epoch: {}".format(steps_per_epoch))

print("Starting training...\n")
history = model.fit_generator(
    generator = train_gen,
    steps_per_epoch = steps_per_epoch,
    epochs = epochs,
    verbose = verbose,
    callbacks = clbacks,
    validation_data = val_gen,
    validation_steps = validation_steps,
    class_weight = class_weights
)
print("Finished training\n")

now = datetime.datetime.now()
loggername = str(now).split(".")[0]
loggername = loggername.replace(":","-")
print('Date and time: {}\n'.format(loggername))

# In[]:
#img = images_train[1]
#a = model.predict(preprocess_input(np.expand_dims(get_image(img), axis=0)))
#print(np.argmax(a))
#print(img)

# In[]:




# In[]:




# In[]:




# In[]:




