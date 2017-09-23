from models.basic.basic_model import BasicModel
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from utils import Constants
from utils.Measures import create_confusion_matrix
from config.onavos_config import OnavosConfig
from utils import Measures
import glob
from tensorflow.python.training import moving_averages
from collections import namedtuple
from datasets.Loader import load_dataset
from scipy.ndimage import imread
import os
import numpy

import numpy

# IMAGENET_RGB_MEAN = numpy.array((124.0, 117.0, 104.0), dtype=numpy.float32) / 255.0
# values from https://github.com/itijyou/ademxapp/blob/0239e6cf53c081b3803ccad109a7beb56e3a386f/iclass/ilsvrc.py
IMAGENET_RGB_MEAN = numpy.array([0.485, 0.456, 0.406], dtype="float32")
IMAGENET_RGB_STD = numpy.array([0.229, 0.224, 0.225], dtype="float32")


def normalize(img, img_mean=IMAGENET_RGB_MEAN, img_std=IMAGENET_RGB_STD):
    if hasattr(img, "get_shape"):
        l = img.get_shape()[-1]
        if img_mean is not None and l != img_mean.size:
            img_mean = numpy.concatenate([img_mean, numpy.zeros(l - img_mean.size, dtype="float32")], axis=0)
        if img_std is not None and l != img_std.size:
            img_std = numpy.concatenate([img_std, numpy.ones(l - img_std.size, dtype="float32")], axis=0)

    if img_mean is not None:
        img -= img_mean
    if img_std is not None:
        img /= img_std
    return img


class Onavos():
    def __init__(self, sess, config):
        # super.__init__(args)
        # self.args = args
        self.config = config
        self.sess = sess
        self.coordinator = tf.train.Coordinator()

        self.valid_data = load_dataset(config, "valid", self.sess, self.coordinator)

        self.regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)
        self.init_input()

        inputs_tensors_dict = self.valid_data.create_input_tensors_dict(self.config.batch_size)
        # inputs and labels are not optional
        self.inputs = inputs_tensors_dict["inputs"]
        self.labels = inputs_tensors_dict["labels"]
        tf.train.start_queue_runners(self.sess)

        self.raw_labels = inputs_tensors_dict.get("raw_labels", None)
        self.index_imgs = inputs_tensors_dict.get("index_imgs", None)
        self.tags = inputs_tensors_dict.get("tags")

        self.build()

        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver(max_to_keep=2)
        if self.config.load_model:
            self.load()

    def save(self):
        self.saver.save(self.sess, self.config.checkpoint_dir)
        print("Model saved")

    def load(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
            print("Model loaded")

    def init_input(self):
        self.img_placeholder = tf.placeholder(tf.float32, shape=(None, 480, 854, 3), name="img_placeholder")
        self.label_placeholder = tf.placeholder(tf.uint8, shape=(None, 480, 854, 1), name="label_placeholder")
        self.tag_placeholder = tf.placeholder(tf.string, shape=(), name="tag_placeholder")
        self.is_training = tf.placeholder(tf.bool)

    def residual_unit2(self, name, inputs, n_convs=2, n_features=None, dilations=None, strides=None,
                       filter_size=None, activation=tf.nn.relu, batch_norm_decay=0.95):
        n_features_inp = int(inputs.get_shape()[-1])
        if dilations is not None:
            assert strides is None
        elif strides is None:
            strides = [[1, 1]] * n_convs
        if filter_size is None:
            filter_size = [[3, 3]] * n_convs
        if n_features is None:
            n_features = n_features_inp
        if not isinstance(n_features, list):
            n_features = [n_features] * n_convs

        with tf.variable_scope(name):
            curr = tf.layers.batch_normalization(inputs, axis=-1, momentum=batch_norm_decay, epsilon=1e-5,
                                                 training=self.is_training, name='bn0')
            if activation is not None:
                curr = activation(curr)

            if strides is None:
                strides_res = [1, 1]
            else:
                strides_res = np.prod(strides, axis=0).tolist()

            if (n_features[-1] != n_features_inp) or (strides_res != [1, 1]):
                if dilations is None:
                    res = tf.layers.conv2d(curr, n_features[-1], 1, strides=strides_res,
                                           kernel_regularizer=self.regularizer,
                                           use_bias=False, padding='SAME', name='conv0')
                else:
                    res = tf.layers.conv2d(curr, n_features[-1], 1, kernel_regularizer=self.regularizer,
                                           use_bias=False, padding='SAME', name='conv0')
            else:
                res = inputs
            if dilations is None:
                curr = tf.layers.conv2d(curr, n_features[0], filter_size[0], strides=strides[0],
                                        kernel_regularizer=self.regularizer, use_bias=False, padding='SAME',
                                        name='conv1')
            else:
                curr = tf.layers.conv2d(curr, n_features[0], filter_size[0], dilation_rate=dilations[0],
                                        kernel_regularizer=self.regularizer, use_bias=False, padding='SAME',
                                        name='conv1')

            for idx in range(1, n_convs):
                curr = tf.layers.batch_normalization(curr, axis=-1, momentum=batch_norm_decay, epsilon=1e-5,
                                                     training=self.is_training, name='bn' + str(idx))

                if activation is not None:
                    curr = activation(curr)

                if dilations is None:
                    curr = tf.layers.conv2d(curr, n_features[idx], filter_size[idx], strides=strides[idx],
                                            kernel_regularizer=self.regularizer, use_bias=False, padding='SAME',
                                            name='conv' + str(idx + 1))
                else:
                    curr = tf.layers.conv2d(curr, n_features[idx], filter_size[idx], dilation_rate=dilations[idx],
                                            kernel_regularizer=self.regularizer, use_bias=False, padding='SAME',
                                            name='conv' + str(idx + 1))

            curr = curr + res
            return curr

    def segmentation_softmax(self, name, inp, targets, n_classes, void_label, filter_size=(1, 1),
                             input_activation=None, dilation=None, resize_targets=False, resize_logits=False, loss="ce",
                             fraction=None):
        n_features_inp = int(inp.get_shape()[-1])
        filter_size = list(filter_size)

        with tf.variable_scope(name):
            if input_activation is not None:
                inp = input_activation(inp)

            if dilation is None:
                y_pred = tf.layers.conv2d(inp, n_classes, filter_size[0], kernel_regularizer=self.regularizer,
                                          padding='SAME', name='conv')
            else:
                y_pred = tf.layers.conv2d(inp, n_classes, filter_size[0], kernel_regularizer=self.regularizer,
                                          dilation_rate=dilation, padding='SAME', name='conv')
            self.outputs = [tf.nn.softmax(y_pred, -1, 'softmax')]

            if resize_targets:
                targets = tf.image.resize_nearest_neighbor(targets, tf.shape(y_pred)[1:3])
            if resize_logits:
                y_pred = tf.image.resize_images(y_pred, tf.shape(targets)[1:3])
            self.pred = tf.argmax(y_pred, axis=-1)

            targets = tf.cast(targets, tf.int64)
            targets = tf.squeeze(targets, axis=3)

            # TODO: Void label is not considered in the iou calculation.
            if void_label is not None:
                # avoid nan by replacing void label by 0
                # note: the loss for these cases is multiplied by 0 below
                void_label_mask = tf.equal(targets, void_label)
                no_void_label_mask = tf.logical_not(void_label_mask)
                targets = tf.where(void_label_mask, tf.zeros_like(targets), targets)
            else:
                no_void_label_mask = None
            self.measures = self.create_measures(self.pred, targets)
            self.loss = self.create_loss(loss, fraction, no_void_label_mask, targets, void_label, y_pred)

    def bootstrapped_ce_loss(self, ce, fraction):
        # only consider k worst pixels (lowest posterior probability) per image
        assert fraction is not None
        batch_size = ce.get_shape().as_list()[0]
        if batch_size is None:
            batch_size = tf.shape(ce)[0]
        k = tf.cast(tf.cast(tf.shape(ce)[1] * tf.shape(ce)[2], tf.float32) * fraction, tf.int32)
        bs_ce, _ = tf.nn.top_k(tf.reshape(ce, shape=[batch_size, -1]), k=k, sorted=False)
        bs_ce = tf.reduce_mean(bs_ce, axis=1)
        bs_ce = tf.reduce_sum(bs_ce, axis=0)
        return bs_ce

    def create_loss(self, loss_str, fraction, no_void_label_mask, targets, void_label, y_pred):
        ce = None
        if "ce" in loss_str:
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=targets, name="ce")

            if void_label is not None:
                mask = tf.cast(no_void_label_mask, tf.float32)
                ce *= mask
        if loss_str == "ce":
            ce = tf.reduce_mean(ce, axis=[1, 2])
            ce = tf.reduce_sum(ce, axis=0)
            loss = ce
        elif loss_str == "bootstrapped_ce":
            bs_ce = self.bootstrapped_ce_loss(ce, fraction)
            loss = bs_ce
        else:
            assert False, "Unknown loss " + loss_str
        return loss

    def create_measures(self, pred, targets, n_classes=2):
        measures = {}
        conf_matrix = tf.py_func(create_confusion_matrix, [pred, targets, n_classes], [tf.int64])
        measures[Constants.CONFUSION_MATRIX] = conf_matrix[0]
        return measures

    def build(self):
        self.conv0 = tf.layers.conv2d(self.inputs, 64, 3, use_bias=False, padding='SAME', name='conv0')

        self.res0 = self.residual_unit2('res0', self.conv0, n_features=128, strides=[[2, 2], [1, 1]])
        self.res1 = self.residual_unit2('res1', self.res0)
        self.res2 = self.residual_unit2('res2', self.res1)

        self.res3 = self.residual_unit2('res3', self.res2, n_features=256, strides=[[2, 2], [1, 1]])
        self.res4 = self.residual_unit2('res4', self.res3)
        self.res5 = self.residual_unit2('res5', self.res4)

        self.res6 = self.residual_unit2('res6', self.res5, n_features=512, strides=[[2, 2], [1, 1]])
        self.res7 = self.residual_unit2('res7', self.res6)
        self.res8 = self.residual_unit2('res8', self.res7)
        self.res9 = self.residual_unit2('res9', self.res8)
        self.res10 = self.residual_unit2('res10', self.res9)
        self.res11 = self.residual_unit2('res11', self.res10)

        self.res12 = self.residual_unit2('res12', self.res11, n_features=[512, 1024], dilations=[1, 2])
        self.res13 = self.residual_unit2('res13', self.res12, n_features=[512, 1024], dilations=[2, 2])
        self.res14 = self.residual_unit2('res14', self.res13, n_features=[512, 1024], dilations=[2, 2])

        self.res15 = self.residual_unit2('res15', self.res14, n_convs=3, n_features=[512, 1024, 2048],
                                         filter_size=[[1, 1], [3, 3], [1, 1]], dilations=[1, 4, 1])

        self.res16 = self.residual_unit2('res16', self.res15, n_convs=3, n_features=[1024, 2048, 4096],
                                         filter_size=[[1, 1], [3, 3], [1, 1]], dilations=[1, 4, 1])

        self.conv1 = tf.layers.batch_normalization(self.res16, axis=-1, momentum=0.95, epsilon=1e-5,
                                                   training=self.is_training, name='conv1/bn')
        self.conv1 = tf.layers.conv2d(tf.nn.relu(self.conv1), 512, 3, dilation_rate=12,
                                      kernel_regularizer=self.regularizer,
                                      padding='SAME', name='conv1')
        self.conv1 = tf.nn.relu(self.conv1)

        self.segmentation_softmax('output', self.conv1, self.labels, 2, None, filter_size=(3, 3),
                                  input_activation=tf.nn.relu,
                                  dilation=12, resize_targets=True, loss="bootstrapped_ce", fraction=0.25)

    def load_weights(self, model_path):
        with open(model_path, 'rb') as output:
            loaded_weights = pickle.load(output, encoding='latin1')
            loaded_model_names = []

            for name, v in loaded_weights.items():
                if name == 'global_step:0':
                    continue
                loaded_model_names.append(name)
            model_weights = {}
            model_names = []

            model_trainables_vars = tf.all_variables()
            for i in range(len(model_trainables_vars)):
                name = model_trainables_vars[i].name
                name = name.replace('kernel', 'W')
                name = name.replace('bn', 'zbn')
                model_names.append(name)
                model_weights[model_trainables_vars[i].name] = model_trainables_vars[i]

            loaded_model_names = sorted(loaded_model_names)
            model_names = sorted(model_names)

            for i in range(len(loaded_model_names)):
                print(i)
                name = model_names[i]
                # if i !=0:
                name = name.replace('W', 'kernel')
                name = name.replace('zbn', 'bn')
                assign_op = model_weights[name].assign(loaded_weights[loaded_model_names[i]])
                self.sess.run(assign_op)
        self.save()
        print("loading weights done")

    def test_seq(self, imgs_path, labels_path):
        measures_accumulated = {}
        num_imgs = 0
        for img_name, label_name in zip(sorted(os.listdir(imgs_path)), sorted(os.listdir(labels_path))):
            num_imgs += 1
            feed_dict = {self.is_training: False}
            measure = self.sess.run([self.measures], feed_dict)
            measures_accumulated = Measures.calc_measures_sum(measures_accumulated, measure[0])
            current_iou = Measures.calc_measures_avg(measure, 1, [255])

            print('image ' + str(num_imgs) + ' finished')
            avg_iou = Measures.calc_measures_avg(measures_accumulated, num_imgs, [255])
            print('current_image iou : ', current_iou, 'avg iou : ', avg_iou)

        measures_accumulated = Measures.calc_measures_avg(measures_accumulated, num_imgs, [255])
        return measures_accumulated

    def test(self, img, label):
        feed_dict = {self.is_training: False}
        measures, img = self.sess.run([self.measures, self.pred], feed_dict)
        # img = self.sess.run([self.pred], feed_dict)
        plt.imsave('test', img[0, :, :], cmap='gray')


def main(_):
    print()
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    sess = tf.Session()

    model = Onavos(sess, OnavosConfig())

    model.load_weights('../model.pkl')
    imgs_path = '/home/gemy/work/datasets/DAVIS/JPEGImages/480p/blackswan/'
    labels_path = '/home/gemy/work/datasets/DAVIS/Annotations/480p/blackswan/'

    model.test_seq(imgs_path, labels_path)
    # measures_accumulated = model.test_seq(imgs_path, labels_path)
    # print('Mean iou : ',measures_accumulated)

    # measures_accumulated = model.test(im.reshape((1, 480, 854, 3)), label.reshape((1, 480, 854, 1)), False)
    # measures_accumulated = Measures.calc_measures_avg(measures_accumulated[0], 1,[255])
    # plt.imsave('test', out[0][0, :, :, 0], cmap='gray')


if __name__ == '__main__':
    tf.app.run(main)
