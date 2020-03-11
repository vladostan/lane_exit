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
segmentation_classes = 5

resize = True
input_shape = (256, 640, 3) if resize else (512, 1280, 3)

backbone = 'resnet18'
random_state = 28
batch_size = 8

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
dataset_dir = "../datasets/supervisely/kisi/"
# subdirs = ["2019-04-24", "2019-05-08", "2019-05-15", "2019-05-20", "2019-05-22", "2019-07-12", "2019-08-23"]
subdirs = ["2019-05-08", "2019-05-15", "2019-05-20", "2019-05-22", "2019-07-12", "2019-08-23"]

obj_class_to_machine_color = dataset_dir + "obj_class_to_machine_color.json"

with open(obj_class_to_machine_color) as json_file:
    object_color = json.load(json_file)

ann_files = []
for subdir in subdirs:
    ann_files += [f for f in glob(dataset_dir + subdir + '/ann/' + '*.json', recursive=True)]
    
print("DATASETS USED: {}".format(subdirs))
print("TOTAL IMAGES COUNT: {}\n".format(len(ann_files)))

# In[]:
def get_image(path, resize = False):
    img = Image.open(path)
    if resize:
        img = img.resize(input_shape[:2][::-1])
    img = np.array(img) 
    return img  

i = 0
with open(ann_files[i]) as json_file:
    data = json.load(json_file)
    tags = data['tags']
    objects = data['objects']
    
img_path = ann_files[i].replace('/ann/', '/img/').split('.json')[0]

print("Images dtype: {}".format(get_image(img_path).dtype))
print("Images shape: {}".format(get_image(img_path, resize = True if resize else False).shape))

# In[]: CREATE MASK
mask = np.zeros((512, 1280), dtype=np.uint8)

for obj in data['objects']:
    classTitle = obj['classTitle']
    if classTitle in ['alternative', 'direct', 'lane_marking', 'crosswalk']:
        points = obj['points']
        exterior = points['exterior']
        interior = points['interior']
        if len(exterior) > 0:
            cv2.fillPoly(mask, np.int32([exterior]), (object_color[classTitle][0]))
        if len(interior) > 0:
            cv2.fillPoly(mask, np.int32([interior]), (0))
    # print(obj, end='\n\n')
    
plt.imshow(mask)

# In[]: Visualise
if visualize:
    x = get_image(img_path, resize = True if resize else False)
    fig, axes = plt.subplots(nrows = 1, ncols = 1)
    axes.imshow(x)
    fig.tight_layout()

# In[]: Prepare for training
val_size = 0.15

print("Train:Val = {}:{}\n".format(1-val_size, val_size))

ann_files_train, ann_files_val = train_test_split(ann_files, test_size=val_size, random_state=random_state)
del(ann_files)

print("Training files count: {}".format(len(ann_files_train)))
print("Validation files count: {}".format(len(ann_files_val)))
        
# In[]: Class weight counting
def cw_count(ann_files):
    print("Class weight calculation started")
    cw_seg = np.zeros(num_classes+1, dtype=np.int64)
    cw_cl = np.zeros(num_classes+1, dtype=np.int64)

    for af in tqdm(ann_files):
        # CLASSIFICATION:
        with open(af) as json_file:
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
               
        # SEGMENTATION:
        label_path = af.replace('/ann/', '/masks_machine/').split('.json')[0]
        l = get_image(label_path, label = True, resize = True if resize else False) == object_color['direct'][0]
        
        for i in range(num_classes+1):
            cw_seg[i] += np.count_nonzero(l==i)
            cw_cl[i] += np.count_nonzero(y1==i)
        
    if sum(cw_cl) == len(ann_files):
        print("Class weights for classification calculated successfully:")
        class_weights_cl = np.median(cw_cl/sum(cw_cl))/(cw_cl/sum(cw_cl))
        for cntr,i in enumerate(class_weights_cl):
            print("Class {} = {}".format(cntr, i))
    else:
        print("Class weights for classification calculation failed")
        
    if sum(cw_seg) == len(ann_files)*input_shape[0]*input_shape[1]:
        print("Class weights for segmentation calculated successfully:")
        class_weights_seg = np.median(cw_seg/sum(cw_seg))/(cw_seg/sum(cw_seg))
        for cntr,i in enumerate(class_weights_seg):
            print("Class {} = {}".format(cntr, i))
    else:
        print("Class weights for segmentation calculation failed")
        
    return class_weights_cl, class_weights_seg
        
if class_weight_counting:
    class_weights_cl_train, class_weights_seg_train = cw_count(ann_files_train)
    if val_size > 0:
        class_weights_cl_val, class_weights_seg_val = cw_count(ann_files_val)
    if test_size > 0:
        class_weights_cl_test, class_weights_seg_test = cw_count(ann_files_test)

# In[]:

    
# In[]: 

    
# In[]:
preprocessing_fn = sm.get_preprocessing(backbone)

train_gen = train_generator(files = ann_files_train, 
                             preprocessing_fn = preprocessing_fn, 
                             aug = aug,
                             batch_size = batch_size)

if val_size > 0:
    val_gen = val_generator(files = ann_files_val, 
                             preprocessing_fn = preprocessing_fn, 
                             batch_size = batch_size_init)

# In[]: Bottleneck
model = sm.Linknet_bottleneck_crop(backbone_name=backbone, input_shape=input_shape, classification_classes=classification_classes, segmentation_classes = segmentation_classes, classification_activation = 'softmax', segmentation_activation='softmax')
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

optimizer = RAdam(lr = 1e-4)
model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=["accuracy"])

# In[]:
monitor = 'val_' if val_size > 0 else ''
    
reduce_lr = callbacks.ReduceLROnPlateau(monitor = monitor+'loss', factor = 0.5, patience = 5, verbose = 1, min_lr = 1e-8)
early_stopper = callbacks.EarlyStopping(monitor = monitor+'loss', patience = 10, verbose = 1)

clbacks = [reduce_lr, early_stopper]

if log:
    csv_logger = callbacks.CSVLogger('logs/{}.log'.format(loggername))
    model_checkpoint = callbacks.ModelCheckpoint('weights/{}.hdf5'.format(loggername), monitor = monitor+'loss', verbose = 1, save_best_only = True, save_weights_only = True)
    clbacks.append(csv_logger)
    clbacks.append(model_checkpoint)

print("Callbacks used:")
for c in clbacks:
    print("{}".format(c))

# In[]: 
steps_per_epoch = len(ann_files_train)//batch_size
validation_steps = len(ann_files_val)//batch_size
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
