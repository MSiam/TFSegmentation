import numpy as np
import h5py
import argparse
import os
from scipy import misc
import matplotlib.pyplot as plt
from tqdm import tqdm

images = ['frankfurt_000001_046272_leftImg8bit.png']
gts = ['frankfurt_000001_046272_gtFine_labelIds.png']

# Data arguments
img_height = 512
img_width = 1024

numpy_images = np.empty((0, img_height, img_width, 3), dtype=np.uint8)
numpy_gts = np.empty((0, img_height, img_width), dtype=np.uint8)

labelID_2_trainID = {0: 19,  # 'unlabeled'
                     1: 19,  # 'ego vehicle'
                     2: 19,  # 'rectification border'
                     3: 19,  # 'out of roi'
                     4: 19,  # 'static'
                     5: 19,  # 'dynamic'
                     6: 19,  # 'ground'
                     7: 0,  # 'road'
                     8: 1,  # 'sidewalk'
                     9: 19,  # 'parking'
                     10: 19,  # 'rail track'
                     11: 2,  # 'building'
                     12: 3,  # 'wall'
                     13: 4,  # 'fence'
                     14: 19,  # 'guard rail'
                     15: 19,  # 'bridge'
                     16: 19,  # 'tunnel'
                     17: 5,  # 'pole'
                     18: 19,  # 'polegroup'
                     19: 6,  # 'traffic light'
                     20: 7,  # 'traffic sign'
                     21: 8,  # 'vegetation'
                     22: 9,  # 'terrain'
                     23: 10,  # 'sky'
                     24: 11,  # 'person'
                     25: 12,  # 'rider'
                     26: 13,  # 'car'
                     27: 14,  # 'truck'
                     28: 15,  # 'bus'
                     29: 19,  # 'caravan'
                     30: 19,  # 'trailer'
                     31: 16,  # 'train'
                     32: 17,  # 'motorcycle'
                     33: 18,  # 'bicycle'
                     -1: 19,  # 'license plate'
                     }


def custom_ignore_labels(img, h, w):
    for i in range(h):
        for j in range(w):
            img[i, j] = labelID_2_trainID[img[i, j]]
    return img


for f in tqdm(images):
    img = misc.imread(f)
    if img.shape != (img_height, img_width, 3):
        print("OOPS " + str(img.shape))
        img = misc.imresize(img, (img_height, img_width))
    img = np.expand_dims(img, axis=0)
    numpy_images = np.append(numpy_images, img, axis=0)

for f in tqdm(gts):
    img = misc.imread(f)
    img = custom_ignore_labels(img, img.shape[0], img.shape[1])
    if img.shape != (img_height, img_width):
        print("OOPS " + str(img.shape))
        img = misc.imresize(img, (img_height, img_width), 'nearest')
    img = np.expand_dims(img, axis=0)
    numpy_gts = np.append(numpy_gts, img, axis=0)

print(numpy_images.shape)
print(numpy_gts.shape)
np.save('debug_frankfurt_000001_x', numpy_images)
np.save('debug_frankfurt_000001_y', numpy_gts)
