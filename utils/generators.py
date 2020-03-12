# -*- coding: utf-8 -*-

import numpy as np
import json
from .augmentators import augment
from .funcs import get_image
from keras.utils import to_categorical

def generator(files, preprocess_input_fn, batch_size = 1, input_shape = (256, 640, 3), resize = True, classification_classes = 2, segmentation_classes = 6, do_aug = False, validate = False):
    
    i = 0
    
    while True:
        
        x_batch = np.zeros((batch_size, input_shape[0], input_shape[1], 3), dtype=np.uint8)
        y1_batch = np.zeros((batch_size), dtype=np.int32)
        y2_batch = np.zeros((batch_size, input_shape[0], input_shape[1]), dtype=np.int32)
        
        for b in range(batch_size):
            
            if i == len(files):
                i = 0
              
            # IMAGE
            x = get_image(path=files[i].replace('/ann/', '/img/').split('.json')[0], input_shape=input_shape, resize=resize)
            
            if do_aug and not validate:
                x = augment(x)
            x_batch[b] = x
            
            # CLASSIFICATION
            with open(files[i]) as json_file:
                data = json.load(json_file)
                tags = data['tags']

            y1 = 0
            if len(tags) > 0:
                for tag in range(len(tags)):
                    tag_name = tags[tag]['name']
                    if tag_name == 'offlane':
                        value = tags[tag]['value']
                        if value == '1':
                            y1 = 1
                            break                    
            y1_batch[b] = y1
        
            # SEGMENTATION
            y2 = get_image(path=files[i].replace('/ann/', '/masks_vlad/').split('.json')[0], input_shape=input_shape, label=True, resize=resize)
            y2_batch[b] = y2

            i += 1
            
        x_batch = preprocess_input_fn(x_batch)
        y1_batch = to_categorical(y1_batch, num_classes=classification_classes).astype(np.int32)
        y2_batch = to_categorical(y2_batch, num_classes=segmentation_classes).astype(np.int32)
    
        y_batch = {'classification_output': y1_batch, 'segmentation_output': y2_batch}
        
        yield (x_batch, y_batch)