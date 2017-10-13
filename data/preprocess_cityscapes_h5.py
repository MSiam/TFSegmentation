import numpy as np
import h5py
import argparse
import os
from scipy import misc
from tqdm import tqdm
import tensorflow as tf
from PIL import Image
import scipy.misc as misc
import pdb
import tables


def write_image_annotation_pairs_to_h5(filename_pairs, h5_filename):
    writer = None
    atom = tables.Int8Atom()
    h5_file = tables.open_file(h5_filename, mode='a')
    array_x = h5_file.create_earray(h5_file.root, 'X', atom, (0, 512, 1024, 3))
    array_y = h5_file.create_earray(h5_file.root, 'Y', atom, (0, 512, 1024))
    h = 512
    w = 1024
    for img_path, annotation_path in tqdm(filename_pairs):
        img = misc.imread(img_path)
        img = misc.imresize(img, (h, w))
        annotation = misc.imread(annotation_path)
        annotation = custom_ignore_labels(annotation)
        annotation = misc.imresize(annotation, (h, w), 'nearest')
        print(img.dtype)
        print(img.shape)
        print(annotation.dtype)
        print(annotation.shape)
        exit(-1)
    writer.close()


def main(args):
    d = args.dir
    train_images_path = args.root + 'images/' + d + '/'
    train_labels_path = args.root + 'labels/' + d + '/'
    custom_read_cityscape(train_images_path, train_labels_path, args)


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


def custom_ignore_labels(img):
    img_temp = np.zeros(img.shape, dtype=np.uint8)
    for k, v in labelID_2_trainID.items():
        img_temp[img == k] = v
    return img_temp


def custom_read_cityscape(path_images, path_labels, args_):
    filename_pairs = []

    # Read all image files in the path_images directory
    for d in os.listdir(path_images):
        for _, dirs, files in os.walk(path_images + '/' + d):
            for f in sorted(files):
                if f.split('.')[-1].lower() == 'png':
                    filename_pairs.append((path_images + '/' + d + '/' + f,
                                           path_labels + d + '/' + f.replace('leftImg8bit', 'gtCoarse_labelIds')))
                else:
                    continue
    write_image_annotation_pairs_to_h5(filename_pairs, args_.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default='/data/cityscapes/',
                        help='path to the dataset root')
    parser.add_argument("--dir", default='train_extra',
                        help='path to the dataset folder')
    parser.add_argument("--out", default='cscapes_train.tfrecords',
                        help='name of output tfrecords file')
    parser.add_argument("--rescale", default=0.25, type=float,
                        help="rescale ratio. eg --rescale 0.5")
    args = parser.parse_args()
    main(args)
