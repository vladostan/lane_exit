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
save_results = False

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

# For each class in classification
TP_0 = 0
FP_0 = 0
FN_0 = 0
TN_0 = 0

TP_1 = 0
FP_1 = 0
FN_1 = 0
TN_1 = 0

TP_2 = 0
FP_2 = 0
FN_2 = 0
TN_2 = 0

TP_3 = 0
FP_3 = 0
FN_3 = 0
TN_3 = 0

TP_mean = 0
FP_mean = 0
FN_mean = 0
TN_mean = 0

mAccuracy_segm = 0
mPrecision_segm = 0
mRecall_segm = 0
mIU_segm = 0
mF1_segm = 0

dlina = len(data_test)
    
for i, data in tqdm(enumerate(data_test)):
        
    timeofday = data['attributes']['timeofday']
    if timeofday == "undefined":
        y1_true = np.int64(0)
    elif timeofday == "daytime":
        y1_true = np.int64(1)
    elif timeofday == "dawn/dusk":
        y1_true = np.int64(2)
    elif timeofday == "night":
        y1_true = np.int64(3)
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
        
    y1_pred = to_categorical(y1_pred, num_classes=classification_classes)
    y1_true = to_categorical(y1_true, num_classes=classification_classes)
    
    for cl in range(classification_classes):
        TP, FP, FN, TN = tpfpfn(y1_pred[cl], y1_true[cl])
        if cl == 0:
            TP_0 += TP
            FP_0 += FP
            FN_0 += FN
            TN_0 += TN
        elif cl == 1:
            TP_1 += TP
            FP_1 += FP
            FN_1 += FN
            TN_1 += TN
        elif cl == 2:
            TP_2 += TP
            FP_2 += FP
            FN_2 += FN
            TN_2 += TN
        elif cl == 3:
            TP_3 += TP
            FP_3 += FP
            FN_3 += FN
            TN_3 += TN
        TP_mean += TP
        FP_mean += FP
        FN_mean += FN
        TN_mean += TN

    y2_true = y2_true.astype('int64')  
    
    mAccuracy_segm += mAccuracy(y2_pred, y2_true, num_classes=segmentation_classes)/dlina
    mPrecision_segm += mPrecision(y2_pred, y2_true, num_classes=segmentation_classes)/dlina
    mRecall_segm += mRecall(y2_pred, y2_true, num_classes=segmentation_classes)/dlina
    mIU_segm += mIU(y2_pred, y2_true, num_classes=segmentation_classes)/dlina
    mF1_segm += mF1(y2_pred, y2_true, num_classes=segmentation_classes)/dlina
   
mAccuracy_0 = Accuracy(TP_0, FP_0, FN_0, TN_0)
mPrecision_0 = Precision(TP_0, FP_0)
mRecall_0 = Recall(TP_0, FN_0)
mIU_0 = IU(TP_0, FP_0, FN_0)
mF1_0 = F1(TP_0, FP_0, FN_0)

mAccuracy_1 = Accuracy(TP_1, FP_1, FN_1, TN_1)
mPrecision_1 = Precision(TP_1, FP_1)
mRecall_1 = Recall(TP_1, FN_1)
mIU_1 = IU(TP_1, FP_1, FN_1)
mF1_1 = F1(TP_1, FP_1, FN_1)

mAccuracy_2 = Accuracy(TP_2, FP_2, FN_2, TN_2)
mPrecision_2 = Precision(TP_2, FP_2)
mRecall_2 = Recall(TP_2, FN_2)
mIU_2 = IU(TP_2, FP_2, FN_2)
mF1_2 = F1(TP_2, FP_2, FN_2)

mAccuracy_3 = Accuracy(TP_3, FP_3, FN_3, TN_3)
mPrecision_3 = Precision(TP_3, FP_3)
mRecall_3 = Recall(TP_3, FN_3)
mIU_3 = IU(TP_3, FP_3, FN_3)
mF1_3 = F1(TP_3, FP_3, FN_3)

print("CLASS 0 accuracy: {}".format(mAccuracy_0))
print("CLASS 0 precision: {}".format(mPrecision_0))
print("CLASS 0 recall: {}".format(mRecall_0))
print("CLASS 0 iu: {}".format(mIU_0))
print("CLASS 0 f1: {}".format(mF1_0))

print("CLASS 1 accuracy: {}".format(mAccuracy_1))
print("CLASS 1 precision: {}".format(mPrecision_1))
print("CLASS 1 recall: {}".format(mRecall_1))
print("CLASS 1 iu: {}".format(mIU_1))
print("CLASS 1 f1: {}".format(mF1_1))

print("CLASS 2 accuracy: {}".format(mAccuracy_2))
print("CLASS 2 precision: {}".format(mPrecision_2))
print("CLASS 2 recall: {}".format(mRecall_2))
print("CLASS 2 iu: {}".format(mIU_2))
print("CLASS 2 f1: {}".format(mF1_2))

print("CLASS 3 accuracy: {}".format(mAccuracy_3))
print("CLASS 3 precision: {}".format(mPrecision_3))
print("CLASS 3 recall: {}".format(mRecall_3))
print("CLASS 3 iu: {}".format(mIU_3))
print("CLASS 3 f1: {}".format(mF1_3))

#mAccuracy_mean = (mAccuracy_0 + mAccuracy_1 + mAccuracy_2 + mAccuracy_3)/4
#mPrecision_mean = (mPrecision_0 + mPrecision_1 + mPrecision_2 + mPrecision_3)/4
#mRecall_mean = (mRecall_0 + mRecall_1 + mRecall_2 + mRecall_2)/4
#mIU_mean = (mIU_0 + mIU_1 + mIU_2 + mIU_3)/4
#mF1_mean = (mF1_0 + mF1_1 + mF1_2 + mF1_3)/4

mAccuracy_mean = Accuracy(TP_mean, FP_mean, FN_mean, TN_mean)
mPrecision_mean = Precision(TP_mean, FP_mean)
mRecall_mean = Recall(TP_mean, FN_mean)
mIU_mean = IU(TP_mean, FP_mean, FN_mean)
mF1_mean = F1(TP_mean, FP_mean, FN_mean)

print("CLASS MEAN accuracy: {}".format(mAccuracy_mean))
print("CLASS MEAN precision: {}".format(mPrecision_mean))
print("CLASS MEAN recall: {}".format(mRecall_mean))
print("CLASS MEAN iu: {}".format(mIU_mean))
print("CLASS MEAN f1: {}".format(mF1_mean))

print("MASK accuracy: {}".format(mAccuracy_segm))
print("MASK precision: {}".format(mPrecision_segm))
print("MASK recall: {}".format(mRecall_segm))
print("MASK iu: {}".format(mIU_segm))
print("MASK f1: {}".format(mF1_segm))