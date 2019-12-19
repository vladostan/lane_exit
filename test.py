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
from PIL import Image
from sklearn.model_selection import train_test_split
import segmentation_models as sm
from keras import optimizers
import cv2

# In[]: Parameters
visualize = True
save_results = True

classification_classes = 1
segmentation_classes = 1

resize = True
input_shape = (256, 640, 3) if resize else (512, 1280, 3)

backbone = 'resnet18'

random_state = 28
batch_size = 1

verbose = 1

weights = "2019-09-30 17-32-13"
#weights = "2019-09-30 17-33-02" 

# In[]:
dataset_dir = "../../../colddata/datasets/supervisely/kamaz/kisi/"

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
preprocessing_fn = sm.get_preprocessing(backbone)

# In[]: Bottleneck
model = sm.Linknet_bottleneck_crop(backbone_name=backbone, input_shape=input_shape, classification_classes=classification_classes, segmentation_classes = segmentation_classes, classification_activation = 'sigmoid', segmentation_activation='sigmoid')
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
if save_results:
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    textPosition           = (5,30)

from metrics import tpfpfn, Accuracy, Precision, Recall, IU, F1

TP_1 = 0
FP_1 = 0
FN_1 = 0
TN_1 = 0

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

dlina = len(ann_files_test)
    
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
    
    if save_results:
        vis_pred = cv2.addWeighted(x_vis,1,cv2.applyColorMap(255//2*np.squeeze(y2_pred > 0.5).astype(np.uint8),cv2.COLORMAP_OCEAN),1,0)

        if y1_pred:
            text = 'Prediction: OFF LANE'
        else:
            text = 'Prediction: IN LANE'
            
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=fontScale, thickness=1)[0]
        box_coords = ((textPosition[0] - 10, textPosition[1] + 10), (textPosition[0] + text_width + 5, textPosition[1] - text_height - 10))
        cv2.rectangle(vis_pred, box_coords[0], box_coords[1], (0,0,0), cv2.FILLED)
        cv2.putText(vis_pred, text, textPosition, font, fontScale, fontColor, lineType)
        
        vis_true = cv2.addWeighted(x_vis,1,cv2.applyColorMap(255//2*y2_true.astype(np.uint8),cv2.COLORMAP_OCEAN),1,0)

        if y1_true:
            text = 'Ground Truth: OFF LANE'
        else:
            text = 'Ground Truth: IN LANE'
            
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=fontScale, thickness=1)[0]
        box_coords = ((textPosition[0] - 10, textPosition[1] + 10), (textPosition[0] + text_width + 5, textPosition[1] - text_height - 10))
        cv2.rectangle(vis_true, box_coords[0], box_coords[1], (0,0,0), cv2.FILLED)
        cv2.putText(vis_true, text, textPosition, font, fontScale, fontColor, lineType)
                 
        if not os.path.exists("results/{}".format(weights)):
            os.mkdir("results/{}".format(weights))
            
        cv2.imwrite("results/{}/{}.png".format(weights, aft.split('/')[-1].split('.')[0]), cv2.cvtColor(np.vstack((vis_pred, vis_true)), cv2.COLOR_BGR2RGB))
    
    y2_true = y2_true.astype('int64')  
    y2_pred = np.squeeze(y2_pred > 0.5).astype('int64')

    TP, FP, FN, TN = tpfpfn(y1_pred, y1_true)
    TP_1 += TP
    FP_1 += FP
    FN_1 += FN
    TN_1 += TN
    
    TP, FP, FN, TN = tpfpfn(y2_pred, y2_true)
    
    mAccuracy_2 += Accuracy(TP, FP, FN, TN)/dlina
    mPrecision_2 += Precision(TP, FP)/dlina
    mRecall_2 += Recall(TP, FN)/dlina
    mIU_2 += IU(TP, FP, FN)/dlina
    mF1_2 += F1(TP, FP, FN)/dlina
    
mAccuracy_1 = Accuracy(TP_1, FP_1, FN_1, TN_1)
mPrecision_1 = Precision(TP_1, FP_1)
mRecall_1 = Recall(TP_1, FN_1)
mIU_1 = IU(TP_1, FP_1, FN_1)
mF1_1 = F1(TP_1, FP_1, FN_1)
    
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