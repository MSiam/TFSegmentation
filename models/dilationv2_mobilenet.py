from models.basic.basic_model import BasicModel
from models.encoders.VGG import VGG16
from models.encoders.mobilenet import MobileNet
from layers.convolution import conv2d_transpose, conv2d, atrous_conv2d, depthwise_separable_conv2d
import numpy as np
import tensorflow as tf
from utils.misc import _debug
import pdb

class DilationV2MobileNet(BasicModel):
    """
    FCN8s with MobileNet as an encoder Model Architecture
    """

    def __init__(self, args, phase=0):
        super().__init__(args, phase=phase)
        # init encoder
        self.encoder = None
        self.wd= self.args.weight_decay

        # init network layers
        self.upscore2 = None
        self.score_feed1 = None
        self.fuse_feed1 = None
        self.upscore4 = None
        self.score_feed2 = None
        self.fuse_feed2 = None
        self.upscore8 = None
        self.targets_size = 8

    def build(self):
        print("\nBuilding the MODEL...")
        self.init_input()
        self.init_network()
        self.init_output()
        self.init_train()
        self.init_summaries()
        print("The Model is built successfully\n")

    def init_input(self):
        with tf.name_scope('input'):
            self.x_pl = tf.placeholder(tf.float32,
                                       [self.args.batch_size, self.params.img_height, self.params.img_width, 3])
            self.y_pl = tf.placeholder(tf.int32, [self.args.batch_size, self.params.img_height//self.targets_size,
                self.params.img_width//self.targets_size])
            print('X_batch shape ', self.x_pl.get_shape().as_list(), ' ', self.y_pl.get_shape().as_list())
            print('Afterwards: X_batch shape ', self.x_pl.get_shape().as_list(), ' ', self.y_pl.get_shape().as_list())

            self.curr_learning_rate = tf.placeholder(tf.float32)
            if self.params.weighted_loss:
                self.wghts = np.zeros((self.args.batch_size, self.params.img_height, self.params.img_width),
                                      dtype=np.float32)
            self.is_training = tf.placeholder(tf.bool)


    def init_network(self):
        """
        Building the Network here
        :return:
        """
        # Init MobileNet as an encoder
        self.encoder = MobileNet(x_input=self.x_pl, num_classes=self.params.num_classes,
                                 pretrained_path=self.args.pretrained_path,
                                 train_flag=self.is_training, width_multipler=1.0, weight_decay=self.args.weight_decay)

        # Build Encoding part
        self.encoder.build()

        # Build Decoding part
        with tf.name_scope('dilation_2'):
            self.conv4_2 = atrous_conv2d('conv_ds_7_dil', self.encoder.conv4_1,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      activation=tf.nn.relu, dilation_rate=2,
                                                      batchnorm_enabled=True, is_training=self.is_training,
                                                      l2_strength=self.wd)
            _debug(self.conv4_2)
            self.conv5_1 = depthwise_separable_conv2d('conv_ds_8_dil', self.conv4_2,
                                                      width_multiplier=self.encoder.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu,
                                                      batchnorm_enabled=True, is_training=self.is_training,
                                                      l2_strength=self.wd)
            _debug(self.conv5_1)
            self.conv5_2 = depthwise_separable_conv2d('conv_ds_9_dil', self.conv5_1,
                                                      width_multiplier=self.encoder.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu,
                                                      batchnorm_enabled=True, is_training=self.is_training,
                                                      l2_strength=self.wd)
            _debug(self.conv5_2)
            self.conv5_3 = depthwise_separable_conv2d('conv_ds_10_dil', self.conv5_2,
                                                      width_multiplier=self.encoder.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu,
                                                      batchnorm_enabled=True, is_training=self.is_training,
                                                      l2_strength=self.wd)
            _debug(self.conv5_3)
            self.conv5_4 = depthwise_separable_conv2d('conv_ds_11_dil', self.conv5_3,
                                                      width_multiplier=self.encoder.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu,
                                                      batchnorm_enabled=True, is_training=self.is_training,
                                                      l2_strength=self.wd)
            _debug(self.conv5_4)
            self.conv5_5 = depthwise_separable_conv2d('conv_ds_12_dil', self.conv5_4,
                                                      width_multiplier=self.encoder.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu,
                                                      batchnorm_enabled=True, is_training=self.is_training,
                                                      l2_strength=self.wd)
            _debug(self.conv5_5)
            self.conv5_6 = atrous_conv2d('conv_ds_13_dil', self.conv5_5,
                                                      num_filters=1024, kernel_size=(3, 3), padding='SAME',
                                                      activation=tf.nn.relu, dilation_rate=4,
                                                      batchnorm_enabled=True, is_training=self.is_training,
                                                      l2_strength=self.wd)
            _debug(self.conv5_6)
            self.conv6_1 = depthwise_separable_conv2d('conv_ds_14_dil', self.conv5_6,
                                                      width_multiplier=self.encoder.width_multiplier,
                                                      num_filters=1024, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu,
                                                      batchnorm_enabled=True, is_training=self.is_training,
                                                      l2_strength=self.wd)
            _debug(self.conv6_1)
            # Pooling is removed.
            self.score_fr = conv2d('conv_1c_1x1_dil', self.conv6_1, num_filters=self.params.num_classes, l2_strength=self.wd,
                                   kernel_size=(1, 1))

            _debug(self.score_fr)
            self.logits= self.score_fr

