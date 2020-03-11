# -*- coding: utf-8 -*-

import numpy as np
from albumentations import (
    OneOf,
    Blur,
    HueSaturationValue,
    MedianBlur,
    CLAHE,
    HorizontalFlip,
    Compose,
    GaussNoise,
    MotionBlur,
    ShiftScaleRotate,
    IAASharpen,
    IAAEmboss,
    RandomBrightnessContrast,
    RandomGamma,
    RGBShift,
    RandomBrightness,
    RandomContrast
)

def augmentator(p=0.5):
    return OneOf([
        Blur(blur_limit=5, p=1.),
        RandomGamma(gamma_limit=(50, 150), p=1.),
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.),
        RGBShift(r_shift_limit=15, g_shift_limit=5, b_shift_limit=15, p=1.),
        RandomBrightness(limit=.25, p=1.),
        RandomContrast(limit=.25, p=1.),
        MedianBlur(blur_limit=5, p=1.),
        CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.)
        ], p=p)
    
def augment(img, p=0.5):
    return augmentator(p=p)(image=img)['image']