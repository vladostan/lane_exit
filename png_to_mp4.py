# -*- coding: utf-8 -*-

import cv2
from glob import glob
from tqdm import tqdm

# In[]:
imgs_dir = "../../../colddata/segmification/res1/"
imgs = [f for f in glob(imgs_dir + '*.png', recursive=True)]
imgs.sort()
print("TOTAL IMAGES COUNT: {}\n".format(len(imgs)))

size = (640, 256)

out = cv2.VideoWriter('../../../colddata/segmification/segmification.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

for img in tqdm(imgs):
    img = cv2.imread(img)
    out.write(img)
out.release()