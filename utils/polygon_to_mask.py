# -*- coding: utf-8 -*-

import json
from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm

# In[]:
dataset_dir = "../../datasets/supervisely/kisi/"
subdirs = ["2019-04-24", "2019-05-08", "2019-05-15", "2019-05-20", "2019-05-22", "2019-07-12", "2019-08-23"]

ann_files = []
for subdir in subdirs[1:]:
    ann_files += [f for f in glob(dataset_dir + subdir + '/ann/' + '*.json', recursive=True)]
    
print("TOTAL FILES COUNT: {}\n".format(len(ann_files)))

# In[]:
input_shape = (512, 1280, 3)

def get_image(path):
    img = Image.open(path)
    img = np.array(img) 
    return img  

# In[]:
color = {'alternative':1, 'direct':2, 'dashed':3, 'solid':4, 'crosswalk':5}
# color = {'alternative':50, 'direct':100, 'dashed':150, 'solid':200, 'crosswalk':250}
classes = ['alternative', 'direct', 'lane_marking', 'crosswalk']
alphabet = "adlcbefghijkmnopqrstuvwxyz_0123456789"

for ann_file in tqdm(ann_files):
    mask = np.zeros((512, 1280, 3), dtype=np.uint8)

    with open(ann_file) as json_file:
        data = json.load(json_file)
        objects = data['objects']
        objects = [obj for obj in objects if obj['classTitle'] in classes] # Remove detection classes
        objects.sort(key = lambda i: [alphabet.index(c) for c in i['classTitle']]) # Sort
        
        for obj in objects:
            classTitle = obj['classTitle']
            c = classTitle
            if classTitle in classes:
                points = obj['points']
                exterior = points['exterior']
                interior = points['interior']
                if classTitle == 'lane_marking':
                    c = 'solid'
                    for t in obj['tags']:
                        if t['name'] == 'dashed':
                            dashed = t['value']
                            if dashed == '1':                                
                                c = 'dashed'

                if len(exterior) > 0:
                    cv2.fillPoly(mask, np.int32([exterior]), (color[c],color[c],color[c]))
                if len(interior) > 0:
                    cv2.fillPoly(mask, np.int32([interior]), (0,0,0))
                    
    split = ann_file.split('/ann/')
    folder = split[0] + '/masks_vlad'
    fname = split[-1].split('.json')[0]
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    plt.imsave(f"{folder}/{fname}", mask)
            
# In[]:
i = 0
img_path = ann_files[i].replace('/ann/', '/img/').split('.json')[0]
x = get_image(img_path)
fig, axes = plt.subplots(nrows = 2, ncols = 1)
axes[0].imshow(x)
axes[1].imshow(mask)
fig.tight_layout()

print(np.unique(mask))

# In[]:





# In[]:





# In[]:





# In[]:





# In[]:





# In[]:





