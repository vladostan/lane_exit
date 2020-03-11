# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image

def get_image(path, input_shape, label = False, resize = False):
    img = Image.open(path)
    if resize:
        img = img.resize(input_shape[:2][::-1], resample=Image.NEAREST)
    img = np.array(img) 
    if label:
        return img[..., 0]
    return img  