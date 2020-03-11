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
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2
from utils.funcs import get_image
from utils.generators import generator
from utils.logger import Logger

# import segmentation_models as sm
# from segmentation_models import Linknet_bottleneck_crop
# from keras import optimizers, callbacks
# from keras_radam import RAdam

# In[]: Parameters
log = False
visualize = True
class_weight_counting = True
aug = True
verbose = 1

classification_classes = 2
segmentation_classes = 6

resize = True
input_shape = (256, 640, 3) if resize else (512, 1280, 3)

backbone = 'resnet18'
lr = 1e-3
random_state = 28
batch_size = 8
val_size = 0.2

weights = None

# In[]: Logger
loggername = str(datetime.datetime.now()).split('.')[0].replace(':','-').replace(' ','_')
print(f"Date and time: {loggername}\n")

if log:
    sys.stdout = Logger()

print(f"""LOG: {log}
AUG: {aug}
LEARNING RATE: {lr}
CLASSIFICATION CLASSES: {classification_classes}
SEGMENTATION CLASSES: {segmentation_classes}
INPUT SHAPE: {input_shape}
BACKBONE: {backbone}
RANDOM STATE: {random_state}
BATCH SIZE: {batch_size}""")

# In[]:
dataset_dir = "../datasets/supervisely/kisi/"
subdirs = ["2019-04-24", "2019-05-08", "2019-05-15", "2019-05-20", "2019-05-22", "2019-07-12", "2019-08-23"]

ann_files = []
for subdir in subdirs[1:]:
    ann_files += [f for f in glob(dataset_dir + subdir + '/ann/' + '*.json', recursive=True)]
    
print(f"DATASETS USED: {subdirs}")
print(f"TOTAL IMAGES COUNT: {len(ann_files)}\n")

# In[]:
i = 0
with open(ann_files[i]) as json_file:
    data = json.load(json_file)
    tags = data['tags']
    objects = data['objects']
    
img_path = ann_files[i].replace('/ann/', '/img/').split('.json')[0]
label_path = ann_files[0].replace('/ann/', '/masks_vlad/').split('.json')[0]

print(f"Images dtype: {get_image(img_path, input_shape).dtype}")
print(f"Labels dtype: {get_image(label_path, input_shape, label = True).dtype}\n")
print(f"Images shape: {get_image(img_path, input_shape, resize = True if resize else False).shape}")
print(f"Labels shape: {get_image(label_path, input_shape, label = True, resize = True if resize else False).shape}\n")

# In[]: Visualise
if visualize:
    x = get_image(img_path, input_shape, resize = True if resize else False)
    y = get_image(label_path, input_shape, label = True, resize = True if resize else False)
    fig, axes = plt.subplots(nrows = 2, ncols = 1)
    axes[0].imshow(x)
    axes[1].imshow(y)
    fig.tight_layout()

# In[]: Prepare for training
print(f"Train:Val = {1-val_size}:{val_size}\n")

ann_files_train, ann_files_val = train_test_split(ann_files, test_size=val_size, random_state=random_state)

print(f"Training files count: {len(ann_files_train)}")
print(f"Validation files count: {len(ann_files_val)}")
        
# In[]: Class weight counting
cw_cl = np.zeros(classification_classes, dtype=np.int64)
cw_seg = np.zeros(segmentation_classes, dtype=np.int64)

print("Class weight calculation started")
for aft in tqdm(ann_files_train):
    with open(aft) as json_file:
        data = json.load(json_file)
        tags = data['tags']

    # CLASSIFICATION:
    y1 = 0
    if len(tags) > 0:
        for tag in range(len(tags)):
            tag_name = tags[tag]['name']
            if tag_name == 'offlane':
                value = tags[tag]['value']
                if value == '1':
                    y1 = 1
                    break
    
    for i in range(classification_classes):
        cw_cl[i] += np.count_nonzero(y1==i)
           
    # SEGMENTATION:
    label_path = aft.replace('/ann/', '/masks_vlad/').split('.json')[0]
    l = get_image(label_path, input_shape, label = True, resize = True if resize else False)
    
    for i in range(segmentation_classes):
        cw_seg[i] += np.count_nonzero(l==i)
        
if sum(cw_cl) == len(ann_files_train):
    print("Class weights for classification calculated successfully:")
    class_weights_cl = np.median(cw_cl/sum(cw_cl))/(cw_cl/sum(cw_cl))
    for cntr,i in enumerate(class_weights_cl):
        print(f"Class {cntr} = {i}")
else:
    print("Class weights for classification calculation failed")
    
if sum(cw_seg) == len(ann_files_train)*input_shape[0]*input_shape[1]:
    print("Class weights for segmentation calculated successfully:")
    class_weights_seg = np.median(cw_seg/sum(cw_seg))/(cw_seg/sum(cw_seg))
    for cntr,i in enumerate(class_weights_seg):
        print(f"Class {cntr} = {i}")
else:
    print("Class weights for segmentation calculation failed")

cw = {"classification_output": class_weights_cl, "segmentation_output": class_weights_seg}
    
# In[]:
preprocessing_fn = sm.get_preprocessing(backbone)

train_gen = generator(files = ann_files_train, 
                             preprocessing_fn = preprocessing_fn, 
                             batch_size = batch_size, 
                             input_shape = input_shape, 
                             resize = resize, 
                             classification_classes = classification_classes, 
                             segmentation_classes = segmentation_classes, 
                             do_aug = True, 
                             validate = False)

val_gen = generator(files = ann_files_val, 
                         preprocessing_fn = preprocessing_fn, 
                         batch_size = batch_size, 
                         input_shape = input_shape, 
                         resize = resize, 
                         classification_classes = classification_classes, 
                         segmentation_classes = segmentation_classes, 
                         do_aug = False, 
                         validate = True)

# In[]: Bottleneck
model = sm.Linknet_bottleneck_crop(backbone_name=backbone, input_shape=input_shape, classification_classes=classification_classes, segmentation_classes=segmentation_classes, classification_activation='softmax', segmentation_activation='softmax')

if weights:
    model.load_weights(weights)

model.summary()

# In[]: 
losses = {
        "classification_output": "binary_crossentropy",
        "segmentation_output": sm.losses.dice_loss
}

loss_weights = {
        "classification_output": 1.0,
        "segmentation_output": 1.0
}

optimizer = RAdam(learning_rate = lr)
model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=["accuracy"])

# In[]:    
reduce_lr = callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 3, verbose = 1, min_lr = 1e-8)
early_stopper = callbacks.EarlyStopping(monitor = 'val_loss', patience = 8, verbose = 1)

clbacks = [reduce_lr, early_stopper]

if log:
    csv_logger = callbacks.CSVLogger('logs/{}.log'.format(loggername))
    model_checkpoint = callbacks.ModelCheckpoint('weights/{}.hdf5'.format(loggername), monitor = 'val_loss', verbose = 1, save_best_only = True, save_weights_only = True)
    clbacks.append(csv_logger)
    clbacks.append(model_checkpoint)

print("Callbacks used:")
for c in clbacks:
    print(f"{c}")

# In[]: 
steps_per_epoch = len(ann_files_train)//batch_size
validation_steps = len(ann_files_val)//batch_size
epochs = 1000

print(f"Steps per epoch: {steps_per_epoch}")

print("Starting training...\n")
history = model.fit_generator(
        generator = train_gen,
        steps_per_epoch = steps_per_epoch,
        epochs = epochs,
        verbose = verbose,
        callbacks = clbacks,
        validation_data = val_gen,
        validation_steps = validation_steps,
        class_weight = cw
)
print("Finished training\n")
print(f"Date and time: {str(datetime.datetime.now()).split('.')[0].replace(':','-').replace(' ','_')}")
