# -*- coding: utf-8 -*-

# In[1]:
import os
import numpy as np
import time
from segmentation_models import Linknet, Linknet_bottleneck_crop
from keras import optimizers
from tqdm import tqdm

# In[]
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1" 

# In[4]:
num_classes = 1
resize = True
input_shape = (256, 640, 3) if resize else (512, 1280, 3)
backbone = 'resnet18'

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

#losses = [dice_coef_binary_loss]

optimizer = optimizers.Adam(lr = 1e-4)

#model = Linknet(backbone_name=backbone, input_shape=input_shape, classes=num_classes, activation='sigmoid')
model = Linknet_bottleneck_crop(backbone_name=backbone, input_shape=input_shape, classes=num_classes, activation='sigmoid')

#model.compile(optimizer=optimizer, loss=losses, metrics=["accuracy"])
model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=["accuracy"])

# In[56]:
model.predict(np.zeros((1,input_shape[0],input_shape[1],3)))

num = 1000

start_time = time.time()

for i in tqdm(range(num)):
    
    model.predict(np.zeros((1,input_shape[0],input_shape[1],3)))
                
print("--- {} seconds ---".format(time.time() - start_time))
print("--- {} fps ---".format(num/(time.time() - start_time)))