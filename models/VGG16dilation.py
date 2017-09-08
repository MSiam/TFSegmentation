from models.basic.basic_model import BasicModel
from models.encoders.VGG import VGG16
from layers.convolution import conv2d_transpose, conv2d, atrous_conv2d
from layers.dense import load_dense_layer

import tensorflow as tf


class VGG16Dilation(BasicModel):
    """
    FCN8s Model Architecture
    """

    def __init__(self, args):
        super().__init__(args)
        # init encoder
        self.encoder = None
        # layers
        self.conv5_3_dil = None
        self.fc6_dil = None
        self.fc7 = None
        self.score_fr = None
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
        with tf.name_scope('dilation_2'):
            self.conv5_3_dil = atrous_conv2d('conv5_3_dil', x=self.encoder.conv5_2, num_filters=512,
                                             kernel_size=(3, 3), dilation_rate=2, activation=tf.nn.relu,
                                             l2_strength=self.encoder.wd, is_training=self.is_training)

            self.fc6_dil = atrous_conv2d('fc6_dil', x=self.conv5_3_dil, num_filters=1024,
                                         kernel_size=(7, 7), dilation_rate=4, activation=tf.nn.relu,
                                         l2_strength=self.encoder.wd, dropout_keep_prob=0.5,
                                         is_training=self.is_training)

            self.fc7 = load_dense_layer(self.encoder.reduced_flag, self.fc6_dil, 'fc7', self.encoder.pretrained_weights,
                                        activation=tf.nn.relu, dropout=0.5,
                                        train=self.is_training, l2_strength=self.encoder.wd)

            self.score_fr = load_dense_layer(self.encoder.reduced_flag, self.fc7, 'score_fr',
                                             self.encoder.pretrained_weights,
                                             num_classes=self.params.num_classes, l2_strength=self.encoder.wd)

        with tf.name_scope('upscore_8s'):
            self.upscore8 = conv2d_transpose('upscore8', x=self.score_fr,
                                             output_shape=self.x_pl.shape.as_list()[0:3] + [self.params.num_classes],
                                             kernel_size=(16, 16), stride=(8, 8), l2_strength=self.encoder.wd)

        self.logits = self.upscore8
