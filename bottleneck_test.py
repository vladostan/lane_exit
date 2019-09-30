# -*- coding: utf-8 -*-

# In[]: Set GPU
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# In[]: Imports
import pickle
import json
from tqdm import tqdm
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
from segmentation_models import Linknet_bottleneck_crop
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
visualize = False

num_classes = 1

resize = True
input_shape = (256, 640, 3) if resize else (512, 1280, 3)

backbone = 'resnet18'

random_state = 28
batch_size = 1

verbose = 1

#weights = "2019-09-27 17-40-50"
#weights = "2019-09-27 17-41-40"
#weights = "2019-09-27 17-44-01"

#weights = "2019-09-30 17-32-13"
weights = "2019-09-30 17-33-02" 
#[0.002347076744235192, 0.0023221724831683378, 2.490426791962551e-05, 0.9947150285427387, 1.0]

# In[]:
dataset_dir = "../../../colddata/datasets/supervisely/kamaz/kisi/"
#subdirs = ["2019-04-24", "2019-05-08", "2019-05-15"]
#subdirs = ["2019-05-20"]
subdirs = ["2019-04-24", "2019-05-08", "2019-05-15", "2019-05-20"]

obj_class_to_machine_color = dataset_dir + "obj_class_to_machine_color.json"

with open(obj_class_to_machine_color) as json_file:
    object_color = json.load(json_file)

ann_files = []
for subdir in subdirs:
    ann_files += [f for f in glob(dataset_dir + subdir + '/ann/' + '*.json', recursive=True)]
    
print("DATASETS USED: {}".format(subdirs))
print("TOTAL IMAGES COUNT: {}\n".format(len(ann_files)))
    
# In[]:
def get_image(path, label = False, resize = False):
    img = Image.open(path)
    if resize:
        img = img.resize(input_shape[:2][::-1])
    img = np.array(img) 
    if label:
        return img[..., 0]
    return img  

with open(ann_files[0]) as json_file:
    data = json.load(json_file)
    tags = data['tags']
    objects = data['objects']
    
img_path = ann_files[0].replace('/ann/', '/img/').split('.json')[0]
label_path = ann_files[0].replace('/ann/', '/masks_machine/').split('.json')[0]

print("Images dtype: {}".format(get_image(img_path).dtype))
print("Labels dtype: {}\n".format(get_image(label_path, label = True).dtype))
print("Images shape: {}".format(get_image(img_path, resize = True if resize else False).shape))
print("Labels shape: {}\n".format(get_image(label_path, label = True, resize = True if resize else False).shape))

# In[]: Prepare for training
#val_size = 0.
#test_size = 0.9999
#
#print("Train:Val:Test split = {}:{}:{}\n".format(1-val_size-test_size, val_size, test_size))
#
#ann_files_train, ann_files_valtest = train_test_split(ann_files, test_size=val_size+test_size, random_state=random_state)
#ann_files_val, ann_files_test = train_test_split(ann_files_valtest, test_size=test_size/(test_size+val_size+1e-8)-1e-8, random_state=random_state)
#del(ann_files_valtest)
#
#print("Training files count: {}".format(len(ann_files_train)))
#print("Validation files count: {}".format(len(ann_files_val)))
#print("Testing files count: {}\n".format(len(ann_files_test)))

with open('pickles/{}.pickle'.format(weights), 'rb') as f:
    ann_files_train = pickle.load(f)
    ann_files_val = pickle.load(f)
    ann_files_test = pickle.load(f)
    
# In[]: 
def predict_generator(files, preprocessing_fn = None, batch_size = 1):
    
    i = 0
    
    while True:
        
        x_batch = np.zeros((batch_size, input_shape[0], input_shape[1], 3), dtype=np.uint8)
        
        for b in range(batch_size):
            
            if i == len(files):
                i = 0
                
            x = get_image(ann_files[i].replace('/ann/', '/img/').split('.json')[0], resize = True if resize else False)
            
            x_batch[b] = x
                
            i += 1
            
        x_batch = preprocessing_fn(x_batch)
            
        yield x_batch
        
        
def val_generator(files, preprocessing_fn = None, batch_size = 1):
    
    i = 0
    
    while True:
        
        x_batch = np.zeros((batch_size, input_shape[0], input_shape[1], 3), dtype=np.uint8)
        y1_batch = np.zeros((batch_size, num_classes), dtype=np.int64)
        y2_batch = np.zeros((batch_size, input_shape[0], input_shape[1]))
        
        for b in range(batch_size):
            
            if i == len(files):
                i = 0
                
            x = get_image(ann_files[i].replace('/ann/', '/img/').split('.json')[0], resize = True if resize else False)
            
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
                        
            y2 = get_image(ann_files[i].replace('/ann/', '/masks_machine/').split('.json')[0], label=True, resize = True if resize else False)
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

eval_gen = val_generator(files = ann_files_test, 
                             preprocessing_fn = preprocessing_fn, 
                             batch_size = batch_size)

# In[]: Bottleneck
model = Linknet_bottleneck_crop(backbone_name=backbone, input_shape=input_shape, classes=num_classes, activation='sigmoid')
model.load_weights('weights/' + weights + '.hdf5')

# In[]: 
from losses import dice_coef_binary_loss

losses = {
        "classification_output": "binary_crossentropy",
        "segmentation_output": dice_coef_binary_loss
}

loss_weights = {
        "classification_output": 1.0,
        "segmentation_output": 1.0
}

optimizer = optimizers.Adam(lr = 1e-4)
model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=["accuracy"])

# In[]:
i = 228
x = get_image(ann_files[i].replace('/ann/', '/img/').split('.json')[0], resize = True if resize else False)
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

if visualize:
    plt.imshow(np.squeeze(y2_pred > 0.5))
    
offlane = np.squeeze(y1_pred) > 0.5

print("OFFLANE PREDICT: {}".format(offlane))
print("OFFLANE GT: {}".format(bool(y1_true)))

# In[]:
#steps = len(ann_files_test)//batch_size
#
#history = model.evaluate_generator(
#        generator = eval_gen,
#        steps = steps,
#        verbose = verbose
#        )
#
#print(history)

#with open('evaluate.pickle', 'wb') as f:
#    pickle.dump(history, f)
    
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
#y_pred = model.predict_generator(
#        generator = predict_gen,
#        steps = steps,
#        verbose = verbose
#        )
#
#with open('predict.pickle', 'wb') as f:
#    pickle.dump(y_pred, f)

# In[]:
#y_pred1 = y_pred[1]
#y_pred2 = y_pred[0]   

# In[]:
import cv2
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (5,30)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

for aft in tqdm(ann_files_test):
    
    x = get_image(aft.replace('/ann/', '/img/').split('.json')[0], resize = True if resize else False)
    x_vis = x.copy()
    x = preprocessing_fn(x)
    y_pred = model.predict(np.expand_dims(x,axis=0))
    y1_pred = y_pred[1]
    y1_pred = np.squeeze(y1_pred) > 0.5
    y2_pred = y_pred[0]
    
    with open(aft) as json_file:
        data = json.load(json_file)
        tags = data['tags']

    y1_true = False
    if len(tags) > 0:
        for tag in range(len(tags)):
            tag_name = tags[tag]['name']
            if tag_name == 'offlane':
                value = tags[tag]['value']
                if value == '1':
                    y1_true = True
                    break
                
    y2_true = get_image(aft.replace('/ann/', '/masks_machine/').split('.json')[0], resize = True if resize else False)
    y2_true = y2_true == object_color['direct'][0]
    y2_true = y2_true[...,0]
    
    vis_pred = cv2.addWeighted(x_vis,1,cv2.applyColorMap(255//2*np.squeeze(y2_pred > 0.5).astype(np.uint8),cv2.COLORMAP_OCEAN),1,0)
    cv2.putText(vis_pred, 'Prediction', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    if y1_pred:
        cv2.putText(vis_pred, 'OFFLANE', (500,30), font, fontScale, (255,0,0), lineType)
    
    vis_true = cv2.addWeighted(x_vis,1,cv2.applyColorMap(255//2*y2_true.astype(np.uint8),cv2.COLORMAP_OCEAN),1,0)
    cv2.putText(vis_true, 'Ground Truth', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    if y1_true:
        cv2.putText(vis_true, 'OFFLANE', (500,30), font, fontScale, (255,0,0), lineType)
     
#    plt.imshow(np.vstack((vis_pred, vis_true)))
        
    if not os.path.exists("results/{}".format(weights)):
        os.mkdir("results/{}".format(weights))
        
    cv2.imwrite("results/{}/{}.png".format(weights, aft.split('/')[-1].split('.')[0]), cv2.cvtColor(np.vstack((vis_pred, vis_true)), cv2.COLOR_BGR2RGB))
    