"""
This file will augment the numpy files it takes and save a new ones
thanks for watching
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from utils.dirs import create_dirs


def plot_imgs(x, folder, mode):
    n = x.shape[0]
    if mode == 'x':
        for i in tqdm(range(n)):
            plt.imsave(folder + str(i) + '.png', x[i])
    elif mode == 'y':
        for i in tqdm(range(n)):
            plt.imsave(folder + str(i) + '.png', x[i])


create_dirs(['../data/data_for_test_n_overfit/x_org', '../data/data_for_test_n_overfit/y_org'])
create_dirs(['../data/data_for_test_n_overfit/x_aug', '../data/data_for_test_n_overfit/y_aug'])

x = np.load('../data/data_for_test_n_overfit/X_train.npy')
y = np.load('../data/data_for_test_n_overfit/Y_train.npy')
print(x.shape)
print(y.shape)
print(x.dtype)
print(y.dtype)

plot_imgs(x, '../data/data_for_test_n_overfit/x_org/', mode='x')
plot_imgs(y, '../data/data_for_test_n_overfit/y_org/', mode='y')

seq = iaa.Sequential([
    iaa.Crop(px=(0, 150)),  # crop images from each side by 0 to 16px (randomly chosen)
])
# Convert the stochastic sequence of augmenters to a deterministic one.
# The deterministic sequence will always apply the exactly same effects to the images.
seq_det = seq.to_deterministic()  # call this for each batch again, NOT only once at the start
x_aug = seq_det.augment_images(x)
y_aug = seq_det.augment_images(y)

seq = iaa.Sequential([
    iaa.Fliplr(1),  # horizontally flip 50% of the images
])
# Convert the stochastic sequence of augmenters to a deterministic one.
# The deterministic sequence will always apply the exactly same effects to the images.
seq_det = seq.to_deterministic()  # call this for each batch again, NOT only once at the start
x_aug = np.append(x_aug, seq_det.augment_images(x), axis=0)
y_aug = np.append(y_aug, seq_det.augment_images(y), axis=0)
seq = iaa.Sequential([
    iaa.Flipud(1),  # horizontally flip 50% of the images
])
seq_det = seq.to_deterministic()  # call this for each batch again, NOT only once at the start
x_aug = np.append(x_aug, seq_det.augment_images(x), axis=0)
y_aug = np.append(y_aug, seq_det.augment_images(y), axis=0)

print(x_aug.shape)
print(y_aug.shape)
print(x_aug.dtype)
print(y_aug.dtype)

plot_imgs(x_aug, '../data/data_for_test_n_overfit/x_aug/', mode='x')
plot_imgs(y_aug, '../data/data_for_test_n_overfit/y_aug/', mode='y')
