from models.basic.basic_model import BasicModel
from models.encoders.resnet_18 import RESNET18
from layers.convolution import conv2d_transpose, conv2d

import tensorflow as tf


class LinkNET(BasicModel):
    def __init__(self, args):
        super().__init__(args)
        # init encoder
        self.encoder = None
        # all layers

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
        self.encoder = RESNET18(x_input=self.x_pl,
                                num_classes=self.params.num_classes,
                                pretrained_path=self.args.pretrained_path,
                                train_flag=self.is_training,
                                weight_decay=self.args.weight_decay)

        # Build Encoding part
        self.encoder.build()

        # Build Decoding part
        with tf.name_scope('upscale_1'):
            self.upscale1 = conv2d_transpose('upscale1', x=self.encoder.conv5_3,
                                             output_shape=self.encoder.conv4_3.shape.as_list()[0:3] + [
                                                 self.encoder.conv5_3.shape.as_list()[3]],
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd)
            self.concat1 = tf.concat([self.upscale1, self.encoder.conv4_3], 3)
            self.expand11 = conv2d('expand1_1', x=self.concat1,
                                   num_filters=self.encoder.conv4_3.shape.as_list()[3], kernel_size=(3, 3),
                                   l2_strength=self.encoder.wd)
            self.expand12 = conv2d('expand1_2', x=self.expand11,
                                   num_filters=self.encoder.conv4_3.shape.as_list()[3], kernel_size=(3, 3),
                                   l2_strength=self.encoder.wd)
        with tf.name_scope('upscale_2'):
            self.upscale2 = conv2d_transpose('upscale2', x=self.expand12,
                                             output_shape=self.encoder.conv3_3.shape.as_list()[0:3] + [
                                                 self.encoder.conv4_3.shape.as_list()[3]],
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd)
            self.concat2 = tf.concat([self.upscale2, self.encoder.conv3_3], 3)
            self.expand21 = conv2d('expand2_1', x=self.concat2,
                                   num_filters=self.encoder.conv3_3.shape.as_list()[3], kernel_size=(3, 3),
                                   l2_strength=self.encoder.wd)
            self.expand22 = conv2d('expand2_2', x=self.expand21,
                                   num_filters=self.encoder.conv3_3.shape.as_list()[3], kernel_size=(3, 3),
                                   l2_strength=self.encoder.wd)
        with tf.name_scope('upscale_3'):
            self.upscale3 = conv2d_transpose('upscale3', x=self.expand22,
                                             output_shape=self.encoder.conv2_2.shape.as_list()[0:3] + [
                                                 self.encoder.conv3_3.shape.as_list()[3]],
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd)
            self.concat3 = tf.concat([self.upscale3, self.encoder.conv2_2], 3)
            self.expand31 = conv2d('expand3_1', x=self.concat3,
                                   num_filters=self.encoder.conv2_2.shape.as_list()[3], kernel_size=(3, 3),
                                   l2_strength=self.encoder.wd)
            self.expand32 = conv2d('expand3_2', x=self.expand31,
                                   num_filters=self.encoder.conv2_2.shape.as_list()[3], kernel_size=(3, 3),
                                   l2_strength=self.encoder.wd)
        with tf.name_scope('upscale_4'):
            self.upscale4 = conv2d_transpose('upscale4', x=self.expand32,
                                             output_shape=self.encoder.conv1_2.shape.as_list()[0:3] + [
                                                 self.encoder.conv2_2.shape.as_list()[3]],
                                             kernel_size=(4, 4), stride=(2, 2), l2_strength=self.encoder.wd)
            self.concat4 = tf.concat([self.upscale4, self.encoder.conv1_2], 3)
            self.expand41 = conv2d('expand4_1', x=self.concat4,
                                   num_filters=self.encoder.conv1_2.shape.as_list()[3], kernel_size=(3, 3),
                                   l2_strength=self.encoder.wd)
            self.expand42 = conv2d('expand4_2', x=self.expand41,
                                   num_filters=self.encoder.conv1_2.shape.as_list()[3], kernel_size=(3, 3),
                                   l2_strength=self.encoder.wd)

        with tf.name_scope('final_score'):
            self.fscore = conv2d('fscore', x=self.expand42,
                                 num_filters=self.params.num_classes, kernel_size=(1, 1),
                                 l2_strength=self.encoder.wd)
        self.logits = self.fscore

    def _decoder_block(self, x, out_channels):
        pass

    def _conv_block(self, x):
        pass

    def _full_conv_block(self, x):
        pass