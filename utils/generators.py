# -*- coding: utf-8 -*-

import numpy as np
import json
from .augmentators import augment, cnnlstm_augment

def train_generator(files, preprocessing_fn = None, aug = False, batch_size = 1):
    
    i = 0
    
    while True:
        
        x_batch = np.zeros((batch_factor*batch_size, input_shape[0], input_shape[1], 3), dtype=np.uint8)
        y1_batch = np.zeros((batch_factor*batch_size, num_classes), dtype=np.int64)
        y2_batch = np.zeros((batch_factor*batch_size, input_shape[0], input_shape[1]))
        
        for b in range(batch_size):
            
            if i == len(files):
                i = 0
                
            x = get_image(ann_files[i].replace('/ann/', '/img/').split('.json')[0], resize = True if resize else False)
            
            with open(ann_files[i]) as json_file:
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
                        
            y2 = get_image(ann_files[i].replace('/ann/', '/masks_machine/').split('.json')[0], label=True, resize = True if resize else False)
            y2 = y2 == object_color['direct'][0]
            
            x_batch[batch_factor*b] = x
            y1_batch[batch_factor*b] = y1
            y2_batch[batch_factor*b] = y2
            
            if aug == 1:
                x2 = augment(x)
                x_batch[batch_factor*b+1] = x2
                y1_batch[batch_factor*b+1] = y1
                y2_batch[batch_factor*b+1] = y2
                
            i += 1
            
        x_batch = preprocessing_fn(x_batch)
        y2_batch = np.expand_dims(y2_batch, axis = -1)
        y2_batch = y2_batch.astype('int64')
    
        y_batch = {'classification_output': y1_batch, 'segmentation_output': y2_batch}
        
        yield (x_batch, y_batch)