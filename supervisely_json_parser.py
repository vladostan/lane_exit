# -*- coding: utf-8 -*-

import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob

# In[]:
dataset_dir = "../../../colddata/datasets/supervisely/kamaz/KIA in summer Innopolis/"

# In[]:
obj_class_to_machine_color = dataset_dir + "obj_class_to_machine_color.json"

with open(obj_class_to_machine_color) as json_file:
    object_color = json.load(json_file)

# In[]:
subdirs = ["2019-04-24", "2019-05-08"]

ann_files = []
for subdir in subdirs:
    ann_files += [f for f in glob(dataset_dir + subdir + '/ann/' + '*.json', recursive=True)]

# In[]:
with open(ann_files[0]) as json_file:
    data = json.load(json_file)
    tags = data['tags']
    objects = data['objects']

# In[]:




# In[]:




