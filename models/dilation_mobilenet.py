from models.basic.basic_model import BasicModel
from models.encoders.VGG import VGG16
from models.encoders.mobilenet import MobileNet
from layers.convolution import conv2d_transpose, conv2d

import tensorflow as tf


class DilationMobileNet(BasicModel):
    """
    FCN8s with MobileNet as an encoder Model Architecture
    """

    def __init__(self, args):
        super().__init__(args)
        # init encoder
        self.encoder = None
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
            self.conv4_3_dil = conv2d('conv4_3_dil', x=self.encoder.conv4_2, num_filters=512,
                                        kernel_size=(3, 3), activation= tf.nn.relu,
                                        l2_strength=self.encoder.wd, is_training=self.is_training )

            self.conv5_1_dil = atrous_conv2d('conv5_1_dil', x=self.conv4_3_dil, num_filters=512,
                                             kernel_size=(3, 3), dilation_rate=2, activation=tf.nn.relu,
                                             l2_strength=self.encoder.wd, is_training=self.is_training)

            self.conv5_2_dil = atrous_conv2d('conv5_2_dil', x=self.conv5_1_dil, num_filters=512,
                                             kernel_size=(3, 3), dilation_rate=2, activation=tf.nn.relu,
                                             l2_strength=self.encoder.wd, is_training=self.is_training)

            self.conv5_3_dil = atrous_conv2d('conv5_3_dil', x=self.conv5_2_dil, num_filters=512,
                                             kernel_size=(3, 3), dilation_rate=2, activation=tf.nn.relu,
                                             l2_strength=self.encoder.wd, is_training=self.is_training)

            self.fc6_dil = atrous_conv2d('fc6_dil', x=self.conv5_3_dil, num_filters=1024,
                                         kernel_size=(7, 7), dilation_rate=4, activation=tf.nn.relu,
                                         l2_strength=self.encoder.wd, dropout_keep_prob=0.5,
                                         is_training=self.is_training)

            self.fc7_dil = conv2d('fc7_dil', x=self.fc6_dil, num_filters=1024,
                                        kernel_size=(1, 1), activation= tf.nn.relu, dropout_keep_prob=0.5,
                                        l2_strength=self.encoder.wd, is_training=self.is_training )

            self.score_fr = conv2d('score_fr_dil', x=self.fc7_dil, num_filters=self.params.num_classes,
                                        kernel_size=(1, 1), l2_strength=self.encoder.wd,
                                        is_training=self.is_training )
