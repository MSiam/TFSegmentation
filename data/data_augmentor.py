"""
This file will augment the numpy files it takes and save a new ones
thanks for watching

link library:
https://github.com/aleju/imgaug
http://readthedocs.org/projects/imgaug/
"""


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
import os

def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)



def plot_imgs(x, folder, mode):
    n = x.shape[0]
    if mode == 'x':
        for i in tqdm(range(n)):
            plt.imsave(folder + str(i) + '.png', x[i])
    elif mode == 'y':
        for i in tqdm(range(n)):
            plt.imsave(folder + str(i) + '.png', x[i])


create_dirs(['full_cityscapes_res/x_org', 'full_cityscapes_res/y_org'])
create_dirs(['full_cityscapes_res/x_aug', 'full_cityscapes_res/y_aug'])

x = np.load('full_cityscapes_res/X_train.npy')
y = np.load('full_cityscapes_res/Y_train.npy')

print(x.shape)
print(y.shape)
print(x.dtype)
print(y.dtype)

#plot_imgs(x, 'full_cityscapes_res/x_org/', mode='x')
#plot_imgs(y, 'full_cityscapes_res/y_org/', mode='y')

x_aug = np.empty([0] + list(x.shape[1:]))
y_aug = np.empty([0] + list(y.shape[1:]))

seq = iaa.Sequential([
    iaa.Fliplr(1),  # horizontally flip 50% of the images
#    iaa.Crop(px=(0, 50)),
])
# Convert the stochastic sequence of augmenters to a deterministic one.
# The deterministic sequence will always apply the exactly same effects to the images.
seq_det = seq.to_deterministic()  # call this for each batch again, NOT only once at the start
x_aug = np.append(x_aug, seq_det.augment_images(x), axis=0)
y_aug = np.append(y_aug, seq_det.augment_images(y), axis=0)

print(x_aug.shape)
print(y_aug.shape)
print(x_aug.dtype)
print(y_aug.dtype)

plot_imgs(x_aug, 'full_cityscapes_aug/x_aug/', mode='x')
plot_imgs(y_aug, 'full_cityscapes_aug/y_aug/', mode='y')

# save the new numpys of the augmented data or append it with the real data
np.save('full_cityscapes_res/x_aug.npy',x_aug)
np.save('full_cityscapes_res/y_aug.npy',y_aug)
