from models.basic.basic_model import BasicModel
from models.encoders.VGG import VGG16
from models.encoders.mobilenet import MobileNet
from layers.convolution import conv2d_transpose, conv2d, atrous_conv2d, depthwise_separable_conv2d

import tensorflow as tf
from utils.misc import _debug

class DilationMobileNet(BasicModel):
    """
    FCN8s with MobileNet as an encoder Model Architecture
    """

    def __init__(self, args):
        super().__init__(args)
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

    def build(self):
        print("\nBuilding the MODEL...")
        self.init_input()
        self.init_network()
        self.init_output()
        self.init_train()
        self.init_summaries()
        print("The Model is built successfully\n")

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
            self.conv5_1 = depthwise_separable_conv2d('conv_ds_8_dil', self.conv4_2, width_multiplier=self.encoder.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1), activation=tf.nn.relu,
                                                      batchnorm_enabled=True, is_training=self.is_training,
                                                      l2_strength=self.wd)
            _debug(self.conv5_1)
            self.conv5_2 = depthwise_separable_conv2d('conv_ds_9_dil', self.conv5_1, width_multiplier=self.encoder.width_multiplier,
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
            self.upscore8 = conv2d_transpose('upscore8', x=self.score_fr,
                                             output_shape=self.x_pl.shape.as_list()[0:3] + [self.params.num_classes],
                                             kernel_size=(16, 16), stride=(8, 8), l2_strength=self.encoder.wd, is_training= self.is_training)
            _debug(self.upscore8)
            self.logits= self.upscore8

