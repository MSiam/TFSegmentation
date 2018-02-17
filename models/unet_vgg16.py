from models.basic.basic_model import BasicModel
from models.encoders.VGG import VGG16
from layers.convolution import conv2d_transpose, conv2d

import tensorflow as tf
#import pdb
from utils.misc import _debug

class UNetVGG16(BasicModel):
    def __init__(self, args, phase=0):
        super().__init__(args, phase=phase)
        # init encoder
        self.encoder = None
        # all layers
        self.upscale1 = None
        self.concat1 = None
        self.expand11 = None
        self.expand12 = None
        self.upscale2 = None
        self.concat2 = None
        self.expand21 = None
        self.expand22 = None
        self.upscale3 = None
        self.concat3 = None
        self.expand31 = None
        self.expand32 = None
        self.upscale4 = None
        self.concat4 = None
        self.expand41 = None
        self.expand42 = None
        self.fscore = None

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

        # Init a VGG16 as an encoder
        self.encoder = VGG16(x_input=self.x_pl,
                             num_classes=self.params.num_classes,
                             pretrained_path=self.args.pretrained_path,
                             train_flag=self.is_training,
                             reduced_flag=False,
                             weight_decay=self.args.weight_decay)

        # Build Encoding part
        self.encoder.build()

        # Build Decoding part
        with tf.name_scope('upscale_1'):
            self.upscale1 = conv2d_transpose('upscale0', x=self.encoder.conv5_3,
                                             output_shape=self.encoder.conv4_3.shape.as_list(),
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd)
            _debug(self.upscale1)
            self.concat1 = tf.add(self.upscale1, self.encoder.conv4_3)
            _debug(self.concat1)
            self.expand11 = conv2d('expand1_1', x=self.concat1,
                                   num_filters=self.encoder.conv4_3.shape.as_list()[3], kernel_size=(3, 3),
                                   l2_strength=self.encoder.wd)
            _debug(self.expand11)
            self.expand12 = conv2d('expand1_2', x=self.expand11,
                                   num_filters=self.encoder.conv4_3.shape.as_list()[3], kernel_size=(3, 3),
                                   l2_strength=self.encoder.wd)
            _debug(self.expand12)
        with tf.name_scope('upscale_2'):
            self.upscale2 = conv2d_transpose('upscale2', x=self.expand12,
                                             output_shape=self.encoder.conv3_3.shape.as_list(),
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd)
            _debug(self.upscale2)
            self.concat2 = tf.add(self.upscale2, self.encoder.conv3_3)
            _debug(self.concat2)
            self.expand21 = conv2d('expand2_1', x=self.concat2,
                                   num_filters=self.encoder.conv3_3.shape.as_list()[3], kernel_size=(3, 3),
                                   l2_strength=self.encoder.wd)
            _debug(self.expand21)
            self.expand22 = conv2d('expand2_2', x=self.expand21,
                                   num_filters=self.encoder.conv3_3.shape.as_list()[3], kernel_size=(3, 3),
                                   l2_strength=self.encoder.wd)
            _debug(self.expand22)
        with tf.name_scope('upscale_3'):
            self.upscale3 = conv2d_transpose('upscale3', x=self.expand22,
                                             output_shape=self.encoder.conv2_2.shape.as_list(),
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd)
            _debug(self.upscale3)
            self.concat3 = tf.add(self.upscale3, self.encoder.conv2_2)
            _debug(self.concat3)
            self.expand31 = conv2d('expand3_1', x=self.concat3,
                                   num_filters=self.encoder.conv2_2.shape.as_list()[3], kernel_size=(3, 3),
                                   l2_strength=self.encoder.wd)
            _debug(self.expand31)
            self.expand32 = conv2d('expand3_2', x=self.expand31,
                                   num_filters=self.encoder.conv2_2.shape.as_list()[3], kernel_size=(3, 3),
                                   l2_strength=self.encoder.wd)
            _debug(self.expand32)
        with tf.name_scope('upscale_4'):
            self.upscale4 = conv2d_transpose('upscale4', x=self.expand32,
                                             output_shape=self.encoder.conv1_2.shape.as_list(),
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd)
            _debug(self.upscale4)
            self.concat4 = tf.add(self.upscale4, self.encoder.conv1_2)
            _debug(self.concat4)
            self.expand41 = conv2d('expand4_1', x=self.concat4,
                                   num_filters=self.encoder.conv1_2.shape.as_list()[3], kernel_size=(3, 3),
                                   l2_strength=self.encoder.wd)
            _debug(self.expand41)
            self.expand42 = conv2d('expand4_2', x=self.expand41,
                                   num_filters=self.encoder.conv1_2.shape.as_list()[3], kernel_size=(3, 3),
                                   l2_strength=self.encoder.wd)
            _debug(self.expand42)
        with tf.name_scope('upscale_5'):
            self.upscale5 = conv2d_transpose('upscale5', x=self.expand42,
                                             output_shape=self.encoder.conv1_1.shape.as_list(),
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd)
            _debug(self.upscale5)
            self.concat5 = tf.add(self.upscale5, self.encoder.conv1_1)
            _debug(self.concat5)
            self.expand51 = conv2d('expand5_1', x=self.concat5,
                                   num_filters=self.encoder.conv1_1.shape.as_list()[3], kernel_size=(3, 3),
                                   l2_strength=self.encoder.wd)
            _debug(self.expand51)
            self.expand52 = conv2d('expand5_2', x=self.expand51,
                                   num_filters=self.encoder.conv1_1.shape.as_list()[3], kernel_size=(3, 3),
                                   l2_strength=self.encoder.wd)
            _debug(self.expand52)

        with tf.name_scope('final_score'):
            self.fscore = conv2d('fscore', x=self.expand52,
                                 num_filters=self.params.num_classes, kernel_size=(1, 1),
                                 l2_strength=self.encoder.wd)
            _debug(self.fscore)

        self.logits = self.fscore
