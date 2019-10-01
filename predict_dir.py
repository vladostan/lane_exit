# -*- coding: utf-8 -*-

# In[]: Set GPU
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# In[]: Imports
from tqdm import tqdm
import matplotlib.pylab as plt
from glob import glob
import numpy as np
from PIL import Image
from segmentation_models.backbones import get_preprocessing
from segmentation_models import Linknet_bottleneck_crop
from keras import optimizers
import cv2

# In[]: Parameters
save_results = True

num_classes = 1

resize = True
input_shape = (256, 640, 3) if resize else (512, 1280, 3)

backbone = 'resnet18'
batch_size = 1
verbose = 1

weights = "2019-09-30 17-32-13"

# In[]:
imgs_dir = "../../../colddata/segmification/im1/"
imgs = [f for f in glob(imgs_dir + '*.png', recursive=True)]
imgs.sort()
print("TOTAL IMAGES COUNT: {}\n".format(len(imgs)))
    
# In[]:
def get_image(path, label = False, resize = False):
    img = Image.open(path)
    if resize:
        img = img.resize(input_shape[:2][::-1])
    img = np.array(img) 
    if label:
        return img[..., 0]
    return img  

print("Images dtype: {}".format(get_image(imgs[0]).dtype))
print("Images shape: {}".format(get_image(imgs[0], resize = True if resize else False).shape))
    
# In[]: Bottleneck
preprocessing_fn = get_preprocessing(backbone)
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
x = get_image(imgs[i], resize = True if resize else False)
x = preprocessing_fn(x)
y_pred = model.predict(np.expand_dims(x,axis=0)) 

# In[]:
if save_results:
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (5,30)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

for img in tqdm(imgs):
    
    x = get_image(img, resize = True if resize else False)
    x_vis = x.copy()
    x = preprocessing_fn(x)
    y_pred = model.predict(np.expand_dims(x,axis=0))
    y1_pred = y_pred[1]
    y1_pred = np.squeeze(y1_pred) > 0.5
    y2_pred = y_pred[0]
    
    if save_results:
        vis_pred = cv2.addWeighted(x_vis,1,cv2.applyColorMap(255//2*np.squeeze(y2_pred > 0.5).astype(np.uint8),cv2.COLORMAP_OCEAN),1,0)
        if y1_pred:
            vis_pred = cv2.addWeighted(vis_pred,1,np.array([255,0,0], dtype = np.uint8)*np.ones_like(vis_pred),0.5,0)
            
        cv2.imwrite("../../../colddata/segmification/res1/{}.png".format(img.split('/')[-1].split('.')[0]), cv2.cvtColor(vis_pred, cv2.COLOR_BGR2RGB))