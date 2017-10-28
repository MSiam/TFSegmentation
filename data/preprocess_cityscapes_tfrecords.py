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

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_image_annotation_pairs_to_tfrecord(filename_pairs, tfrecords_filename):
    """Writes given image/annotation pairs to the tfrecords file.
    The function reads each image/annotation pair given filenames
    of image and respective annotation and writes it to the tfrecord
    file.
    Parameters
    ----------
    filename_pairs : array of tuples (img_filepath, annotation_filepath)
        Array of tuples of image/annotation filenames
    tfrecords_filename : string
        Tfrecords filename to write the image/annotation pairs
    """
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    h=512; w= 1024;
    for img_path, annotation_path in filename_pairs:
        print('processing image ', img_path)
#        img = np.array(Image.open(img_path))
        img= misc.imread(img_path)
        img = misc.imresize(img, (h, w))

#        annotation = np.array(Image.open(annotation_path))
        annotation= misc.imread(annotation_path)
        annotation = custom_ignore_labels(annotation, img.shape[0], img.shape[1])
        annotation = misc.imresize(annotation, (h, w), 'nearest')
        # Unomment this one when working with surgical data
        # annotation = annotation[:, :, 0]

        # The reason to store image sizes was demonstrated
        # in the previous example -- we have to know sizes
        # of images to later read raw serialized string,
        # convert to 1d array and convert to respective
        # shape that image used to have.
        height = img.shape[0]
        width = img.shape[1]

        img_raw = img.tostring()
        annotation_raw = annotation.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw),
            'mask_raw': _bytes_feature(annotation_raw)}))

        writer.write(example.SerializeToString())

    writer.close()

def main(args):
    d= args.dir
    train_images_path = args.root + 'images/'+d+'/'
    train_labels_path = args.root + 'labels/'+d+'/'
    custom_read_cityscape(train_images_path, train_labels_path, args, split='d')

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
    img_temp= np.zeros(img.shape, dtype=np.uint8)
    for k, v in labelID_2_trainID.items():
        img_temp[img==k] = v
    return img_temp


def custom_read_cityscape(path_images, path_labels, args_, split='train'):
    filename_pairs= []
    root = path_images
    # Read all image files in the path_images directory
    for d in os.listdir(path_images):
        for _, dirs, files in os.walk(path_images+'/'+d):
            for f in sorted(files):
                if f.split('.')[-1].lower() == 'png':
                    filename_pairs.append( (path_images+'/'+d+'/'+f,\
                        path_labels+d+'/'+f.replace('leftImg8bit','gtCoarse_labelIds')) )
                else:
                    continue
    write_image_annotation_pairs_to_tfrecord(filename_pairs, args_.out)



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
