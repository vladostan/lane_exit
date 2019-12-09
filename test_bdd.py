# -*- coding: utf-8 -*-

# In[]: Set GPU
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# In[]: Imports
import json
from tqdm import tqdm
import matplotlib.pylab as plt
import numpy as np
from keras.utils import to_categorical
from PIL import Image
import segmentation_models as sm
from keras import optimizers
import cv2

# In[]: Parameters
visualize = False
save_results = True

if save_results:
    save_num = 100

classification_classes = 4
segmentation_classes = 3

input_shape = (320, 640, 3)

backbone = 'resnet50'

random_state = 28
batch_size = 1

verbose = 1

#weights = "2019-12-05 16-25-45"
weights = "2019-12-06 10-28-15"

# In[]:
dataset_dir = "../../datasets/bdd/"

ann_file_test = dataset_dir + "labels/" + 'bdd100k_labels_images_val.json'  
    
# In[]:
def get_image(path):
    img = Image.open(path)
    img = img.resize((640,360))
    img = np.array(img)
    return img[20:-20]
    
with open(ann_file_test) as json_file:
    data_test = json.load(json_file)
        
# In[]:
preprocessing_fn = sm.get_preprocessing(backbone)

# In[]: Bottleneck
model = sm.Linknet_bottleneck_crop(backbone_name=backbone, input_shape=input_shape, classification_classes=classification_classes, segmentation_classes = segmentation_classes, classification_activation = 'softmax', segmentation_activation='softmax')
model.load_weights('weights/' + weights + '.hdf5')

# In[]: 
losses = {
        "classification_output": "categorical_crossentropy",
        "segmentation_output": sm.losses.dice_loss
}

loss_weights = {
        "classification_output": 1.0,
        "segmentation_output": 1.0
}

optimizer = optimizers.Adam(lr = 1e-4)
model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=["accuracy"])

# In[]:
i = 100

data = data_test[i]
y1_gt = data['attributes']['timeofday']              
name = data['name']
x = get_image(dataset_dir + 'images/val/' + name)
x = preprocessing_fn(x)
y2_gt = get_image(dataset_dir + 'drivable_maps/labels/val/' + name.split('.jpg')[0] + "_drivable_id.png")

y_pred = model.predict(np.expand_dims(x, axis=0))

# In[]
y1_pred = np.argmax(y_pred[1])
y2_pred = np.argmax(np.squeeze(y_pred[0]), axis=-1)

if visualize:
    plt.imshow(y2_pred)
    
tod_classes = ["undefined", "daytime", "dawn/dusk", "night"]

print("TIME OF DAY PREDICT: {}".format(tod_classes[y1_pred]))
print("TIME OF DAY GT: {}".format(y1_gt))  

# In[]:
if save_results:
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    textPosition           = (5,30)

from metrics import mAccuracy, mPrecision, mRecall, mIU, mF1, tpfpfn, Accuracy, Precision, Recall, IU, F1

mAccuracy_1 = 0
mPrecision_1 = 0
mRecall_1 = 0
mIU_1 = 0
mF1_1 = 0

mAccuracy_2 = 0
mPrecision_2 = 0
mRecall_2 = 0
mIU_2 = 0
mF1_2 = 0

dlina = len(data_test)
    
for i, data in tqdm(enumerate(data_test)):
    
    timeofday = data['attributes']['timeofday']
    if timeofday == "undefined":
        y1_true = 0
    elif timeofday == "daytime":
        y1_true = 1
    elif timeofday == "dawn/dusk":
        y1_true = 2
    elif timeofday == "night":
        y1_true = 3
    else:
        raise ValueError("Impossible value for time of day class")
                
    name = data['name']
        
    x = get_image(dataset_dir + 'images/val/' + name)
    x_vis = x.copy()
    x = preprocessing_fn(x)
    
    y_pred = model.predict(np.expand_dims(x, axis=0))    
    y1_pred = np.argmax(y_pred[1])
    y2_pred = np.argmax(np.squeeze(y_pred[0]), axis=-1)
    
    y2_true = get_image(dataset_dir + 'drivable_maps/labels/val/' + name.split('.jpg')[0] + "_drivable_id.png")
    
    if save_results and i%save_num == 0:
        # PREDICTION
        vis_pred = cv2.addWeighted(x_vis,1,cv2.applyColorMap(y2_pred.astype(np.uint8)*127,cv2.COLORMAP_OCEAN),1,0)
        
        text = 'Prediction: {}'.format(tod_classes[y1_pred])
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=fontScale, thickness=1)[0]
        box_coords = ((textPosition[0] - 10, textPosition[1] + 10), (textPosition[0] + text_width + 5, textPosition[1] - text_height - 10))

        cv2.rectangle(vis_pred, box_coords[0], box_coords[1], (0,0,0), cv2.FILLED)
        cv2.putText(vis_pred, text, textPosition, font, fontScale, fontColor, lineType)
        
        if visualize:
            plt.imshow(vis_pred)
        
        # GROUND TRUTH
        vis_true = cv2.addWeighted(x_vis,1,cv2.applyColorMap(y2_true.astype(np.uint8)*127,cv2.COLORMAP_OCEAN),1,0)
        
        text = 'Ground Truth: {}'.format(tod_classes[y1_true])
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=fontScale, thickness=1)[0]
        box_coords = ((textPosition[0] - 10, textPosition[1] + 10), (textPosition[0] + text_width + 5, textPosition[1] - text_height - 10))
        
        cv2.rectangle(vis_true, box_coords[0], box_coords[1], (0,0,0), cv2.FILLED)
        cv2.putText(vis_true, text, textPosition, font, fontScale, fontColor, lineType)
        
        if visualize:
            plt.imshow(vis_true)
                 
        if not os.path.exists("results/{}".format(weights)):
            os.mkdir("results/{}".format(weights))
            
        cv2.imwrite("results/{}/{}.png".format(weights, name.split('.jpg')[0]), cv2.cvtColor(np.vstack((vis_pred, vis_true)), cv2.COLOR_BGR2RGB))
    
    y1_true = to_categorical(y1_true, num_classes=classification_classes)
    y1_true = y1_true.astype('int64')  
    
    y1_pred = to_categorical(y1_pred, num_classes=classification_classes)
    y1_pred = y1_pred.astype('int64')  

    TP, FP, FN, TN = tpfpfn(y1_pred, y1_true)
    mAccuracy_1 += Accuracy(TP, FP, FN, TN)/dlina
    mPrecision_1 += Precision(TP, FP)/dlina
    mRecall_1 += Recall(TP, FN)/dlina
    mIU_1 += IU(TP, FP, FN)/dlina
    mF1_1 += F1(TP, FP, FN)/dlina

    y2_true = y2_true.astype('int64')  
    
    mAccuracy_2 += mAccuracy(y2_pred, y2_true)/dlina
    mPrecision_2 += mPrecision(y2_pred, y2_true)/dlina
    mRecall_2 += mRecall(y2_pred, y2_true)/dlina
    mIU_2 += mIU(y2_pred, y2_true)/dlina
    mF1_2 += mF1(y2_pred, y2_true)/dlina
   
print("CLASS accuracy: {}".format(mAccuracy_1))
print("CLASS precision: {}".format(mPrecision_1))
print("CLASS recall: {}".format(mRecall_1))
print("CLASS iu: {}".format(mIU_1))
print("CLASS f1: {}".format(mF1_1))

print("MASK accuracy: {}".format(mAccuracy_2))
print("MASK precision: {}".format(mPrecision_2))
print("MASK recall: {}".format(mRecall_2))
print("MASK iu: {}".format(mIU_2))
print("MASK f1: {}".format(mF1_2))