# -*- coding: utf-8 -*-

import numpy as np
from glob import glob
from tqdm import tqdm
import json

# In[]:
dataset_dir = "../../datasets/bdd/"

ann_file_test = dataset_dir + "labels/" + 'bdd100k_labels_images_val.json'  

# In[]:
scenes = {}
timeofdays = {}
weathers = {}


with open(ann_file_test) as json_file:
    data = json.load(json_file)
    
    for d in data:
        attributes = d['attributes']
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





