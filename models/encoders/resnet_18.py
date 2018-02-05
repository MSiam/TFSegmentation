import tensorflow as tf
import numpy as np

from layers.utils import variable_summaries, variable_with_weight_decay
from utils.misc import timeit
from utils.misc import _debug
# import torchfile
import pickle
import pdb

class RESNET18:
    """
    RESNET 18 Encoder class
    """

    def __init__(self, x_input,
                 num_classes,
                 pretrained_path,
                 train_flag,
                 bias=-1,
                 weight_decay=5e-4,
                 test_classification=False):
        """

        :param x_input: Input Images to the RESNET Encoder
        :param num_classes:
        :param pretrained_path:
        :param train_flag:
        :param weight_decay:
        """

        # Load pretrained path
        if pretrained_path.split('.')[-1]=='npy':
            self.pretrained_weights = np.load(pretrained_path)
        elif pretrained_path.split('.')[-1]=='pkl':
            with open(pretrained_path, 'rb') as ff:
               self.pretrained_weights = pickle.load(ff, encoding='latin1')

        print('pretrained weights dictionary loaded from disk')

        # init parameters and input
        self.x_input = x_input
        self.num_classes = num_classes
        self.train_flag = train_flag
        self.wd = weight_decay
        self.bias = bias
        self.use_bias = True
        if self.bias == -1:
            self.use_bias = False

        self.test_classification = test_classification

        # All layers
        self.resnet_mean = None
        self.resnet_std = None
        self.x_preprocessed = None
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.conv4 = None
        self.conv5 = None
        self.score = None

        # These feed layers are for the decoder
        self.feed1 = None
        self.feed2 = None
        self.encoder_1 = None
        self.encoder_2 = None
        self.encoder_3 = None
        self.encoder_4 = None

    def build(self):
        """
        Build the RESNET model using loaded weights
        """

        print("Building the RESNET..")

        # Convert RGB to BGR
        with tf.name_scope('Pre_Processing'):
            self.x_preprocessed = self.x_input * (1.0 / 255.0)
#            self.x_preprocessed= self.x_input
            stat= torchfile.load('stat.t7')
            self.resnet_mean= stat.transpose(1,2,0)
#            self.resnet_mean = tf.constant([0.2869, 0.3251, 0.2839], dtype=tf.float32)
            self.x_preprocessed = (self.x_preprocessed - self.resnet_mean) #/ self.resnet_std
#            red, green, blue = tf.split(self.x_preprocessed, num_or_size_splits=3, axis=3)
#            self.x_preprocessed = tf.concat([blue,green,red], 3)

        # These variables to keep track of what i do
        # filters = [64, 64, 128, 256, 512]
        # kernels = [7, 3, 3, 3, 3]
        # strides = [2, 0, 2, 2, 2]
        tf.add_to_collection('debug_layers', self.x_preprocessed)

        with tf.variable_scope('conv1_x'):
            print('Building unit: conv1')
            self.conv1 = self._conv('conv1', self.x_preprocessed, padding= [[0,0],[3,3],[3,3],[0,0]],
                                    num_filters=64, kernel_size=(7, 7), stride=(2, 2), l2_strength=self.wd,
                                    bias=self.bias)

            self.conv1 = self._bn('bn1', self.conv1)

            self.conv1 = self._relu('relu1', self.conv1)
            _debug(self.conv1)
            self.conv1= tf.pad(self.conv1, tf.constant([[0,0],[1,1],[1,1],[0,0]]), "CONSTANT")
            self.conv1 = tf.nn.max_pool(self.conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',
                                        name='max_pool1')
            _debug(self.conv1)
            print('conv1-shape: ' + str(self.conv1.shape.as_list()))

        with tf.variable_scope('conv2_x'):
            self.conv2 = self._residual_block('conv2_1', self.conv1, 64)
            _debug(self.conv2)
            self.conv2 = self._residual_block('conv2_2', self.conv2, 64)
            _debug(self.conv2)

        with tf.variable_scope('conv3_x'):
            self.conv3 = self._residual_block('conv3_1', self.conv2, 128, pool_first=True, strides=2)
            _debug(self.conv3)
            self.conv3 = self._residual_block('conv3_2', self.conv3, 128)
            _debug(self.conv3)

        with tf.variable_scope('conv4_x'):
            self.conv4 = self._residual_block('conv4_1', self.conv3, 256, pool_first=True, strides=2)
            _debug(self.conv4)
            self.conv4 = self._residual_block('conv4_2', self.conv4, 256)
            _debug(self.conv4)

        with tf.variable_scope('conv5_x'):
            self.conv5 = self._residual_block('conv5_1', self.conv4, 512, pool_first=True, strides=2)
            _debug(self.conv5)
            self.conv5 = self._residual_block('conv5_2', self.conv5, 512)
            _debug(self.conv5)

        if self.test_classification:
            with tf.variable_scope('logits'):
                print('Building unit: logits')
                self.score = tf.reduce_mean(self.conv5, axis=[1, 2])
                self.score = self._fc('logits_dense', self.score, output_dim=self.num_classes, l2_strength=self.wd)
                print('logits-shape: ' + str(self.score.shape.as_list()))

        self.feed1 = self.conv4
        self.feed2 = self.conv3

        self.encoder_1 = self.conv2
        self.encoder_2 = self.conv3
        self.encoder_3 = self.conv4
        self.encoder_4 = self.conv5
        print("\nEncoder RESNET is built successfully\n\n")

    @timeit
    def load_pretrained_weights(self, sess):
        print("Loading pretrained weights of resnet18")
        all_vars = tf.trainable_variables()
        all_vars += tf.get_collection('mu_sigma_bn')
        for v in all_vars:
            if v.op.name in self.pretrained_weights.keys():
                assign_op = v.assign(self.pretrained_weights[v.op.name])
                sess.run(assign_op)
                print(v.op.name + " - loaded successfully")
        print("All pretrained weights of resnet18 is loaded")

    def _residual_block(self, name, x, filters, pool_first=False, strides=1, dilation=1):
        print('Building residual unit: %s' % name)
        with tf.variable_scope(name):
            # get input channels
            in_channel = x.shape.as_list()[-1]

            # Shortcut connection
            shortcut = tf.identity(x)

            if pool_first:
                if in_channel == filters:
                    if strides == 1:
                        shortcut = tf.identity(x)
                    else:
                        shortcut= tf.pad(x, tf.constant([[0,0],[1,1],[1,1],[0,0]]), "CONSTANT")
                        shortcut = tf.nn.max_pool(shortcut, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
                else:
                    shortcut = self._conv('shortcut_conv', x, padding='VALID',
                                          num_filters=filters, kernel_size=(1, 1), stride=(strides, strides),
                                          bias=self.bias)
            else:
                if dilation != 1:
                    shortcut = self._conv('shortcut_conv', x, padding='VALID',
                                          num_filters=filters, kernel_size=(1, 1), dilation=dilation, bias=self.bias)

            # Residual
            x = self._conv('conv_1', x, padding=[[0,0],[1,1],[1,1],[0,0]],
                           num_filters=filters, kernel_size=(3, 3), stride=(strides, strides), bias=self.bias)
            x = self._bn('bn_1', x)
            x = self._relu('relu_1', x)
            x = self._conv('conv_2', x, padding=[[0,0],[1,1],[1,1],[0,0]],
                           num_filters=filters, kernel_size=(3, 3), bias=self.bias)
            x = self._bn('bn_2', x)

            # Merge
            x = x + shortcut
            x = self._relu('relu_2', x)

            print('residual-unit-%s-shape: ' % name + str(x.shape.as_list()))

            return x

    @staticmethod
    def _conv(name, x, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
              initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, dilation=1.0, bias=-1):

        with tf.variable_scope(name):
            stride = [1, stride[0], stride[1], 1]
            kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], num_filters]

            w = variable_with_weight_decay(kernel_shape, initializer, l2_strength)

            variable_summaries(w)
            if dilation > 1:
                conv = tf.nn.atrous_conv2d(x, w, dilation, padding)
            else:
                if type(padding)==type(''):
                    conv = tf.nn.conv2d(x, w, stride, padding)
                else:
                    conv = tf.pad(x, padding, "CONSTANT")
                    conv = tf.nn.conv2d(conv, w, stride, padding='VALID')

            if bias != -1:
                bias = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(bias))

                variable_summaries(bias)
                conv = tf.nn.bias_add(conv, bias)

            tf.add_to_collection('debug_layers', conv)

            return conv

    @staticmethod
    def _relu(name, x):
        with tf.variable_scope(name):
            return tf.nn.relu(x)

    @staticmethod
    def _fc(name, x, output_dim=128,
            initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):

        with tf.variable_scope(name):
            n_in = x.get_shape()[-1].value

            w = variable_with_weight_decay([n_in, output_dim], initializer, l2_strength)

            variable_summaries(w)

            if isinstance(bias, float):
                bias = tf.get_variable("biases", [output_dim], tf.float32, tf.constant_initializer(bias))

            variable_summaries(bias)

            output = tf.nn.bias_add(tf.matmul(x, w), bias)

            return output

    def _bn(self, name, x):
        with tf.variable_scope(name):
            moving_average_decay = 0.9
            decay = moving_average_decay

            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])

            mu = tf.get_variable('mu', batch_mean.shape, dtype=tf.float32,
                                 initializer=tf.zeros_initializer(), trainable=False)
            tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, mu)
            tf.add_to_collection('mu_sigma_bn', mu)
            sigma = tf.get_variable('sigma', batch_var.shape, dtype=tf.float32,
                                    initializer=tf.ones_initializer(), trainable=False)
            tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, sigma)
            tf.add_to_collection('mu_sigma_bn', sigma)
            beta = tf.get_variable('beta', batch_mean.shape, dtype=tf.float32,
                                   initializer=tf.zeros_initializer())
            gamma = tf.get_variable('gamma', batch_var.shape, dtype=tf.float32,
                                    initializer=tf.ones_initializer())

            # BN when training
            update = 1.0 - decay
            update_mu = mu.assign_sub(update * (mu - batch_mean))
            update_sigma = sigma.assign_sub(update * (sigma - batch_var))
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)

            mean, var = tf.cond(self.train_flag, lambda: (batch_mean, batch_var), lambda: (mu, sigma))
            bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)

            tf.add_to_collection('debug_layers', bn)

            return bn
