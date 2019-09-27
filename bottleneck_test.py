# -*- coding: utf-8 -*-

# In[]: Set GPU
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# In[]: Imports
import pickle
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
from segmentation_models import Linknet, Linknet_notop, Linknet_bottleneck, Linknet_bottleneck_crop
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
num_classes = 1
input_shape = (512, 512, 3)
backbone = 'resnet18'

batch_size = 1
verbose = 1

# In[]:
dataset_dir = "../../../colddata/datasets/supervisely/kamaz/kisi/"
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
    img = img[:,384:896]
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

# In[]: Prepare for training
test_size = 0.2

print("Train:test split = {}:{}\n".format(1-test_size, test_size))

ann_files_train, ann_files_test = train_test_split(ann_files, test_size=test_size, random_state=1)

print("Training files count: {}".format(len(ann_files_train)))
print("Testing files count: {}\n".format(len(ann_files_test)))

# In[]: 
def predict_generator(files, preprocessing_fn = None, batch_size = 1):
    
    i = 0
    
    while True:
        
        x_batch = np.zeros((batch_size, input_shape[0], input_shape[1], 3), dtype=np.uint8)
        
        for b in range(batch_size):
            
            if i == len(files):
                i = 0
                
            x = get_image(ann_files[i].replace('/ann/', '/img/').split('.json')[0])
            x_batch[b] = x
                
            i += 1
            
        x_batch = preprocessing_fn(x_batch)
        
        yield x_batch
        
def evaluate_generator(files, preprocessing_fn = None, batch_size = 1):
    
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

predict_gen = predict_generator(files = ann_files_test, 
                             preprocessing_fn = preprocessing_fn, 
                             batch_size = batch_size)

eval_gen = evaluate_generator(files = ann_files_test, 
                             preprocessing_fn = preprocessing_fn, 
                             batch_size = batch_size)

# In[]: Bottleneck
weights = '2019-09-18 10-46-35.hdf5'
model = Linknet_bottleneck_crop(backbone_name=backbone, input_shape=input_shape, classes=num_classes, activation='sigmoid')
model.load_weights('weights/' + weights)

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
i = 5
x = get_image(ann_files[i].replace('/ann/', '/img/').split('.json')[0])
x = preprocessing_fn(x)
y_pred = model.predict(np.expand_dims(x,axis=0))

with open(ann_files[i]) as json_file:
    data = json.load(json_file)
    tags = data['tags']

y1_true = 0
if len(tags) > 0:
    for tag in range(len(tags)):
        tag_name = tags[tag]['name']
        if tag_name == 'offlane':
            value = tags[tag]['value']
            if value == '1':
                y1_true = 1
                break

# In[]
y1_pred = y_pred[1]
y2_pred = y_pred[0]

plt.imshow(np.squeeze(y2_pred > 0.5))
offlane = np.squeeze(y1_pred) > 0.5
print(offlane)

# In[]:
steps = len(ann_files_test)//batch_size

history = model.evaluate_generator(
        generator = eval_gen,
        steps = steps,
        verbose = verbose
        )

with open('evaluate.pickle', 'wb') as f:
    pickle.dump(history, f)
    
# In[]:
#[0.008236413411275073,
# 0.008131222682190941,
# 0.0001051907178693822,
# 0.997653000838273,
# 1.0]
    
#['loss',
# 'segmentation_output_loss',
# 'classification_output_loss',
# 'segmentation_output_acc',
# 'classification_output_acc']
    
# In[]:
y_pred = model.predict_generator(
        generator = predict_gen,
        steps = steps,
        verbose = verbose
        )

with open('predict.pickle', 'wb') as f:
    pickle.dump(y_pred, f)

# In[]:
y_pred1 = y_pred[1]
y_pred2 = y_pred[0]    
    
    
    