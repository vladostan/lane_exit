# coding: utf-8

# In[]: Set GPU
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# In[]: Imports
import matplotlib.pylab as plt
from glob import glob
import numpy as np
import datetime
import sys
import keras
from keras.utils import to_categorical
from PIL import Image
from sklearn.model_selection import train_test_split
from segmentation_models.backbones import get_preprocessing
from segmentation_models import Linknet, Linknet_notop
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
verbose = 2
visualize = False
class_weight_counting = True
doaug = True
batch_size = 1
num_classes = 3
input_shape = (512, 512, 3)
backbone = 'resnet18'

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

# In[]: Read images and masks
SOURCE_IMAGES = os.path.abspath('../datasets/kisi/')

DATASETS = ["2019-04-24", "2019-05-08"]

print("Datasets used: {}\n".format(DATASETS))

images = []
labels = []

for ds in DATASETS:
    il = len(images)
    ll = len(labels)
    images.extend(glob(os.path.join(SOURCE_IMAGES + "/" + ds + "/" + "img", "*.png")))
    print("{} images in {} dataset".format(len(images)-il, ds))
    labels.extend(glob(os.path.join(SOURCE_IMAGES + "/" + ds + "/" + "masks_machine", "*.png")))
    print("{} labels in {} dataset\n".format(len(labels)-ll, ds))

images.sort()
labels.sort()

print("Total images count: {}".format(len(images)))
print("Total labels count: {}\n".format(len(labels)))

# In[]: Read images and labels from file
def get_image(path, label = False):
    img = Image.open(path)
    img = img.resize((input_shape[1],input_shape[0]))
    img = np.array(img) 
    if label:
        return img[...,0]
    return img    

print("Images dtype: {}".format(get_image(images[0]).dtype))
print("Labels dtype: {}\n".format(get_image(labels[0], label=True).dtype))

# In[]: Visualise
if visualize:
    i = 28
    x = get_image(images[i])
    y = get_image(labels[i], label = True)
    fig, axes = plt.subplots(nrows = 2, ncols = 1)
    axes[0].imshow(x)
    axes[1].imshow(y)
    fig.tight_layout()

# In[]: Prepare for training
test_size = 0.

print("Train:test split = {}:{}\n".format(1-test_size, test_size))

images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=test_size, random_state=1)

print("Training images count: {}".format(len(images_train)))
print("Training labels count: {}\n".format(len(labels_train)))
print("Testing images count: {}".format(len(images_test)))
print("Testing labels count: {}\n".format(len(labels_test)))

# In[]: Class weight counting
if class_weight_counting:    
    cw = np.zeros(num_classes, dtype=np.int64)

    for lt in labels_train:
        l = get_image(lt, label=True)
        
        for i in range(num_classes):
            cw[i] += np.count_nonzero(l==i)
        
    if sum(cw) == len(labels_train)*input_shape[0]*input_shape[1]:
        print("Class weights calculated successfully:")
        class_weights = np.median(cw/sum(cw))/(cw/sum(cw))
        for cntr,i in enumerate(class_weights):
            print("Class {} = {}".format(cntr, i))
    else:
        print("Class weights calculation failed")
            
# In[]: Augmentor
if doaug:
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
    
    def augment(image, aug=aug):
        augmented = aug(image=image)
        image_augmented = augmented['image']
        return image_augmented

# In[]: Custom generator
def custom_generator(images_path, labels_path, preprocessing_fn = None, doaug = False, batch_size = 1, validation = False):
    i = 0
    
    while True:
        
        if validation or not doaug:
	        x_batch = np.zeros((batch_size, input_shape[0], input_shape[1], 3))
	        y_batch = np.zeros((batch_size, input_shape[0], input_shape[1]))
        else:
            x_batch = np.zeros((2*batch_size, input_shape[0], input_shape[1], 3))
            y_batch = np.zeros((2*batch_size, input_shape[0], input_shape[1]))
        
        for b in range(batch_size):
            
            if i == len(images_path):
                i = 0
                
            x = get_image(images_path[i])
            y = get_image(labels_path[i], label=True)
            
            if validation or not doaug:
                x_batch[b] = x
                y_batch[b] = y
            else:
                x2 = augment(x)
                x_batch[2*b] = x
                x_batch[2*b+1] = x2
                y_batch[2*b] = y
                y_batch[2*b+1] = y
                
            i += 1
            
        x_batch = preprocessing_fn(x_batch)
        y_batch = to_categorical(y_batch, num_classes=num_classes)
        y_batch = y_batch.astype('int64')
    
        yield (x_batch, y_batch)
           
# In[]: Initialize custom generator
preprocessing_fn = get_preprocessing(backbone)

train_gen = custom_generator(images_path = images_train, 
                             labels_path = labels_train, 
                             preprocessing_fn = preprocessing_fn, 
                             doaug = doaug,
                             batch_size = batch_size)

# In[]: Define segmentation model
num_classes = 2

from keras.models import Model

input, segmentation_model = Linknet_notop(backbone_name=backbone, input_shape=input_shape)

classification_model = SEResNet50(input_shape=input_shape, weights='imagenet', classes=num_classes, include_top=False)

x = classification_model.output(segmentation_model)
x = keras.layers.GlobalAveragePooling2D()(classification_model.output)
output = keras.layers.Dense(num_classes, activation='softmax')(x)
classification_model = keras.models.Model(inputs=[classification_model.input], outputs=[output])

model = Model(input, classification_model)

# In[]: Define classification model
classification_model = SEResNet50(input_shape=input_shape, weights='imagenet', classes=num_classes, include_top=False)
x = keras.layers.GlobalAveragePooling2D()(classification_model.output)
output = keras.layers.Dense(num_classes, activation='sigmoid')(x)
classification_model = keras.models.Model(inputs=[classification_model.input], outputs=[output])

classification_model.summary()

inputs = Input(shape=inputShape)

# In[]:
model = FashionNet.build(96, 96, numCategories=len(categoryLB.classes_), numColors=len(colorLB.classes_), finalAct="softmax")
 
losses = {
        "segmentation_output": [dice_coef_multiclass_loss],
        "classification_output": "binary_crossentropy",
}

lossWeights = {
        "segmentation_output": 1.0, 
        "classification_output": 1.0
        }
 
# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
metrics=["accuracy"])

learning_rate = 1e-4
optimizer = optimizers.Adam(lr = learning_rate)

losses = ['categorical_crossentropy']
metrics = ['accuracy']

print("Optimizer: {}, learning rate: {}, loss: {}, metrics: {}\n".format(optimizer, learning_rate, losses, metrics))

model.compile(optimizer = optimizer, loss = losses, metrics = metrics)

# In[]: Compile model
learning_rate = 1e-4
optimizer = optimizers.Adam(lr = learning_rate)

losses = [dice_coef_multiclass_loss]
metrics = ['categorical_accuracy']

model.compile(optimizer = optimizer, loss = losses, metrics = metrics)
print("\nOptimizer: {}\nLearning rate: {}\nLoss: {}\nMetric: {}\n".format(optimizer, learning_rate, losses, metrics))

# In[]: Initialize callbacks
reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor = 0.5, patience = 5, verbose = 1, min_lr = 1e-8)
early_stopper = callbacks.EarlyStopping(monitor='loss', patience = 10, verbose = 1)
clbacks = [reduce_lr, early_stopper]

if log:
    csv_logger = callbacks.CSVLogger('logs/{}.log'.format(loggername))
    model_checkpoint = callbacks.ModelCheckpoint('weights/{}.hdf5'.format(loggername), monitor = 'loss', verbose = 1, save_best_only = True, save_weights_only = True)
    clbacks.append(csv_logger)
    clbacks.append(model_checkpoint)

print("Callbacks used:")
for c in clbacks:
    print("{}".format(c))

# In[]: Starting training
steps_per_epoch = len(images_train)//batch_size
validation_steps = len(images_test)//batch_size
epochs = 1000

print("\nSteps per epoch: {}".format(steps_per_epoch))
print("Validation steps: {}\n".format(validation_steps))

print("Starting training...")
if class_weight_counting:
    history = model.fit_generator(
        generator = train_gen,
        steps_per_epoch = steps_per_epoch,
        epochs = epochs,
        verbose = verbose,
        callbacks = clbacks,
        class_weight = class_weights
    )
else:
    history = model.fit_generator(
        generator = train_gen,
        steps_per_epoch = steps_per_epoch,
        epochs = epochs,
        verbose = verbose,
        callbacks = clbacks
    )
print("\nFinished training...")

now = datetime.datetime.now()
loggername = str(now).split(".")[0]
loggername = loggername.replace(":","-")
print('Date and time: {}\n'.format(loggername))