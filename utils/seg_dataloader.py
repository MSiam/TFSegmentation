import pdb
import numpy as np
import tensorflow as tf
import random

from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.contrib.data import Iterator
#import cv2
#import matplotlib.pyplot as plt
import scipy

class SegDataLoader(object):
    def __init__(self, main_dir, batch_size, resize_shape, crop_shape, paths_file, buffer_size=100, split='train'):
        self.main_dir= main_dir
        self.batch_size= batch_size
        self.resize_shape= resize_shape
        self.crop_shape= crop_shape
        self.buffer_size= buffer_size
        self.paths_file= paths_file

        self.imgs_files= []
        self.labels_files= []

        # Read image and label paths from file and fill in self.images, self.labels
        self.parse_file(self.paths_file)
        self.shuffle_lists()

        img= convert_to_tensor(self.imgs_files, dtype= dtypes.string)
        label= convert_to_tensor(self.labels_files, dtype= dtypes.string)
        data_tr = tf.data.Dataset.from_tensor_slices((img, label))
        self.data_len= len(self.imgs_files)

        if split == 'train':
            data_tr = data_tr.map(self.parse_train, num_parallel_calls=16)#, num_threads=8, output_buffer_size=100*self.batch_size)
        else:
            data_tr = data_tr.map(self.parse_val,num_parallel_calls=16)#, num_threads=8, output_buffer_size=100*self.batch_size)

        data_tr= data_tr.shuffle(buffer_size)
        self.data_tr= data_tr.batch(batch_size)

    def shuffle_lists(self):
        imgs= self.imgs_files
        labels= self.labels_files

        permutation= np.random.permutation(len(self.imgs_files))
        self.imgs_files= []
        self.labels_files= []
        for i in permutation:
            self.imgs_files.append(imgs[i])
            self.labels_files.append(labels[i])

    def parse_train(self, im_path, label_path):
        # Load image
        img= tf.read_file(im_path)
        img= tf.image.decode_png(img, channels=3)
        last_image_dim = tf.shape(img)[-1]

        # Load label
        label= tf.read_file(label_path)
        label= tf.image.decode_png(label, channels=1)

        # Scale
        img = tf.image.resize_images(img, self.resize_shape, method=tf.image.ResizeMethod.BICUBIC)
        label = tf.image.resize_images(label, self.resize_shape, method= tf.image.ResizeMethod.NEAREST_NEIGHBOR)

#        # combine input and label
#        label = tf.cast(label, dtype=tf.float32)
#        combined = tf.concat([img, label], 2)
#
#        # flipping
#        combined= tf.image.random_flip_left_right(combined)
#
#        # cropping
#        combined_crop = tf.random_crop(combined,[self.crop_shape[0],self.crop_shape[1],4]) # TODO: Make cropping size a variable
#        img, label = (combined_crop[:, :, :last_image_dim], combined_crop[:, :, last_image_dim:])
#        label = tf.cast(label, dtype=tf.uint8)
#        img.set_shape((self.crop_shape[0], self.crop_shape[1], 3))
#        label.set_shape((self.crop_shape[0], self.crop_shape[1], 1))
        return img, label

    def parse_val(self, im_path, label_path):
        # Load image
        img= tf.read_file(im_path)
        img= tf.image.decode_png(img, channels=3)
        last_image_dim = tf.shape(img)[-1]

        # Load label
        label= tf.read_file(label_path)
        label= tf.image.decode_png(label, channels=1)

        # Scale
        img = tf.image.resize_images(img, self.resize_shape, method=tf.image.ResizeMethod.BICUBIC)
        label = tf.image.resize_images(label, self.resize_shape, method= tf.image.ResizeMethod.NEAREST_NEIGHBOR)

#        label = tf.cast(label, dtype=tf.float32)
#        combined = tf.concat([img, label], 2)
#        c1= tf.image.crop_to_bounding_box(combined, 0, 0, self.crop_shape[0], self.crop_shape[1])
#        c2= tf.image.crop_to_bounding_box(combined, 0, self.crop_shape[1], self.crop_shape[0], self.crop_shape[1])
#        imgs= tf.stack([c1,c2],axis=0)
#        img, label = (imgs[:,:, :, :3], imgs[:,:, :, 3:])
#        label = tf.cast(label, dtype=tf.uint8)
#        img.set_shape((2,self.crop_shape[0], self.crop_shape[1], 3))
#        label.set_shape((2,self.crop_shape[0], self.crop_shape[1], 1))
        return img, label

    def parse_file(self, path):
        ff= open(path, 'r')
        for line in ff:
            tokens= line.strip().split(' ')
            self.imgs_files.append(self.main_dir+tokens[0])
            tokens[1]= tokens[1].replace('labelIds', 'labelIds_proc')
            self.labels_files.append(self.main_dir+tokens[1])

    def print_files(self):
        for x, y in zip(self.imgs_files, self.labels_files):
            print(x, y)

if __name__=="__main__":

    config= tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session= tf.Session(config=config)

    with tf.device('/cpu:0'):
        #segdl= SegDataLoader('/home/eren/Data/Cityscapes/', 10, (512,1024), (512,512), 'train.txt')
        segdl= SegDataLoader('/home/eren/Data/Cityscapes/', 10, (512,1024), (512,512), 'val.txt', split='val')
        iterator = Iterator.from_structure(segdl.data_tr.output_types, segdl.data_tr.output_shapes)
        next_batch= iterator.get_next()

        training_init_op = iterator.make_initializer(segdl.data_tr)
        session.run(training_init_op)

    for i in range(10):
       img_batch, label_batch = session.run(next_batch)
#       print(img_batch)
#       img_batch= np.asarray(img_batch,dtype=np.uint8)
#       plt.imshow(label_batch[0,0,:,:,0]);plt.show()
#       plt.imshow(img_batch[0,0,:,:,:]);plt.show()


