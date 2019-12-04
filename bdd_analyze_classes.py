# -*- coding: utf-8 -*-

import numpy as np
from glob import glob
from tqdm import tqdm
import json

# In[]:
dataset_dir = "../../datasets/bdd/bdd100k/"

ann_files_train_val = [f for f in glob(dataset_dir + "labels/100k/train/" + '*.json', recursive=True)]
ann_files_train_val.sort()

# In[]:
scenes = {}
timeofdays = {}
weathers = {}

for ann in tqdm(ann_files_train_val):
    
#    print(ann)
    with open(ann) as json_file:
        try:
            data = json.load(json_file)
            
            attributes = data['attributes']
            scene = attributes['scene']
            timeofday = attributes['timeofday']
            weather = attributes['weather']
            
            if scene not in scenes:
                scenes[scene] = 1
            else:
                scenes[scene] += 1
                
            if timeofday not in timeofdays:
                timeofdays[timeofday] = 1
            else:
                timeofdays[timeofday] += 1          
                
            if weather not in weathers:
                weathers[weather] = 1
            else:
                weathers[weather] += 1    
        except:
            print(ann)

# In[]:
print(scenes)
print(timeofdays)
print(weathers)

# In[]:
#['city street', 'highway', 'residential', 'parking lot', 'undefined', 'tunnel', 'gas stations']
#['daytime', 'dawn/dusk', 'night', 'undefined']
#['clear', 'rainy', 'undefined', 'snowy', 'overcast', 'partly cloudy', 'foggy']

# In[]:





# In[]:





# In[]:





# In[]:





# In[]:





