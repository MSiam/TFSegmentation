from layers.dense import load_dense_layer
from layers.convolution import load_conv_layer
import numpy as np
import tensorflow as tf


class VGG16:
    """
    VGG 16 Encoder class
    """

    VGG_MEAN = [103.939, 116.779, 123.68]

    def __init__(self, x_input,
                 num_classes,
                 pretrained_path,
                 train_flag,
                 reduced_flag=False,
                 weight_decay=5e-4):
        """

        :param x_input: Input to the VGG Encoder
        :param num_classes:
        :param pretrained_path: Path of the numpy which contain the weights
        :param train_flag: Flag of training or inference
        :param reduced_flag:
        :param weight_decay:
        """

        # Load pretrained path
        self.pretrained_weights = np.load(pretrained_path, encoding='latin1').item()
        print('pretrained weights loaded')

        # init parameters and input
        self.x_input = x_input
        self.num_classes = num_classes
        self.train_flag = train_flag
        self.reduced_flag = reduced_flag
        self.wd = weight_decay

        # All layers
        self.conv1_1 = None
        self.conv1_2 = None

        self.conv2_1 = None
        self.conv2_2 = None

        self.conv3_1 = None
        self.conv3_2 = None
        self.conv3_3 = None

        self.conv4_1 = None
        self.conv4_2 = None
        self.conv4_3 = None

        self.conv5_1 = None
        self.conv5_2 = None
        self.conv5_3 = None

        self.fc6 = None
        self.fc7 = None
        self.score_fr = None

        # These feed layers are for the decoder
        self.feed1 = None
        self.feed2 = None

    def build(self):
        """
        Build the VGG model using loaded weights
        """

        print("Building the VGG..")

        # Convert RGB to BGR
        with tf.name_scope('Pre_Processing'):
            red, green, blue = tf.split(self.x_input, num_or_size_splits=3, axis=3)
            preprocessed_input = tf.concat([
                blue - VGG16.VGG_MEAN[0],
                green - VGG16.VGG_MEAN[1],
                red - VGG16.VGG_MEAN[2],
            ], 3)

        self.conv1_1 = load_conv_layer(preprocessed_input, 'conv1_1', self.pretrained_weights, l2_strength=self.wd)
        self.conv1_2 = load_conv_layer(self.conv1_1, 'conv1_2', self.pretrained_weights, pooling=True,
                                       l2_strength=self.wd)

        self.conv2_1 = load_conv_layer(self.conv1_2, 'conv2_1', self.pretrained_weights, l2_strength=self.wd)
        self.conv2_2 = load_conv_layer(self.conv2_1, 'conv2_2', self.pretrained_weights, pooling=True,
                                       l2_strength=self.wd)

        self.conv3_1 = load_conv_layer(self.conv2_2, 'conv3_1', self.pretrained_weights, l2_strength=self.wd)
        self.conv3_2 = load_conv_layer(self.conv3_1, 'conv3_2', self.pretrained_weights, l2_strength=self.wd)
        self.conv3_3 = load_conv_layer(self.conv3_2, 'conv3_3', self.pretrained_weights, pooling=True,
                                       l2_strength=self.wd)

        self.conv4_1 = load_conv_layer(self.conv3_3, 'conv4_1', self.pretrained_weights, l2_strength=self.wd)
        self.conv4_2 = load_conv_layer(self.conv4_1, 'conv4_2', self.pretrained_weights, l2_strength=self.wd)
        self.conv4_3 = load_conv_layer(self.conv4_2, 'conv4_3', self.pretrained_weights, pooling=True,
                                       l2_strength=self.wd)

        self.conv5_1 = load_conv_layer(self.conv4_3, 'conv5_1', self.pretrained_weights, l2_strength=self.wd)
        self.conv5_2 = load_conv_layer(self.conv5_1, 'conv5_2', self.pretrained_weights, l2_strength=self.wd)
        self.conv5_3 = load_conv_layer(self.conv5_2, 'conv5_3', self.pretrained_weights, pooling=True,
                                       l2_strength=self.wd)

        self.fc6 = load_dense_layer(self.reduced_flag, self.conv5_3, 'fc6', self.pretrained_weights,
                                    activation=tf.nn.relu, dropout=0.5,
                                    train=self.train_flag, l2_strength=self.wd)
        self.fc7 = load_dense_layer(self.reduced_flag, self.fc6, 'fc7', self.pretrained_weights, activation=tf.nn.relu,
                                    dropout=0.5,
                                    train=self.train_flag, l2_strength=self.wd)
        self.score_fr = load_dense_layer(self.reduced_flag, self.fc7, 'score_fr', self.pretrained_weights,
                                         num_classes=self.num_classes, l2_strength=self.wd)

        self.feed1 = self.conv4_3
        self.feed2 = self.conv3_3

        print("\nEncoder VGG is built successfully\n\n")
