import numpy as np
import tensorflow as tf
from layers.convolution import depthwise_separable_conv2d, conv2d
from layers.pooling import avg_pool_2d
from layers.dense import dense


class MobileNet:
    """
    MobileNet Class
    """

    def __init__(self, x_input,
                 num_classes,
                 pretrained_path,
                 train_flag,
                 width_multipler=1.0,
                 weight_decay=5e-4):
        self.pretrained_weights = np.load(pretrained_path, encoding='latin1').item()
        print('pretrained weights loaded')

        # init parameters and input
        self.x_input = x_input
        self.num_classes = num_classes
        self.train_flag = train_flag
        self.wd = weight_decay
        self.width_multiplier = width_multipler

        # All layers
        self.conv1_1 = None

        self.conv2_1 = None
        self.conv2_2 = None

        self.conv3_1 = None
        self.conv3_2 = None

        self.conv4_1 = None
        self.conv4_2 = None

        self.conv5_1 = None
        self.conv5_2 = None
        self.conv5_3 = None
        self.conv5_4 = None
        self.conv5_5 = None
        self.conv5_6 = None

        self.conv6_1 = None
        self.flattened = None

        self.score_fr = None

        # These feed layers are for the decoder
        self.feed1 = None
        self.feed2 = None

    def build(self):
        self.encoder_build()
        self.load()

    def encoder_build(self):
        print("Building the MobileNet..")
        with tf.variable_scope('mobilenet_encoder'):
            self.conv1_1 = conv2d('conv_1', self.x_input, num_filters=int(round(32 * self.width_multiplier)),
                                  kernel_size=(3, 3),
                                  padding='SAME', stride=(2, 2), activation=tf.nn.relu, batchnorm_enabled=True,
                                  is_training=self.train_flag, l2_strength=self.wd)

            self.conv2_1 = depthwise_separable_conv2d('conv_ds_2', self.conv1_1, width_multiplier=self.width_multiplier,
                                                      num_filters=64, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self.conv2_2 = depthwise_separable_conv2d('conv_ds_3', self.conv2_1, width_multiplier=self.width_multiplier,
                                                      num_filters=128, kernel_size=(3, 3), padding='SAME',
                                                      stride=(2, 2),
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)

            self.conv3_1 = depthwise_separable_conv2d('conv_ds_4', self.conv2_2, width_multiplier=self.width_multiplier,
                                                      num_filters=128, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1),
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self.conv3_2 = depthwise_separable_conv2d('conv_ds_5', self.conv3_1, width_multiplier=self.width_multiplier,
                                                      num_filters=256, kernel_size=(3, 3), padding='SAME',
                                                      stride=(2, 2),
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)

            self.conv4_1 = depthwise_separable_conv2d('conv_ds_6', self.conv3_2, width_multiplier=self.width_multiplier,
                                                      num_filters=256, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1),
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self.conv4_2 = depthwise_separable_conv2d('conv_ds_7', self.conv4_1, width_multiplier=self.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(2, 2),
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)

            self.conv5_1 = depthwise_separable_conv2d('conv_ds_8', self.conv4_2, width_multiplier=self.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1),
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self.conv5_2 = depthwise_separable_conv2d('conv_ds_9', self.conv5_1, width_multiplier=self.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1),
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self.conv5_3 = depthwise_separable_conv2d('conv_ds_10', self.conv5_2,
                                                      width_multiplier=self.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1),
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self.conv5_4 = depthwise_separable_conv2d('conv_ds_11', self.conv5_3,
                                                      width_multiplier=self.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1),
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self.conv5_5 = depthwise_separable_conv2d('conv_ds_12', self.conv5_4,
                                                      width_multiplier=self.width_multiplier,
                                                      num_filters=512, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1),
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)
            self.conv5_6 = depthwise_separable_conv2d('conv_ds_13', self.conv5_5,
                                                      width_multiplier=self.width_multiplier,
                                                      num_filters=1024, kernel_size=(3, 3), padding='SAME',
                                                      stride=(2, 2),
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)

            self.conv6_1 = depthwise_separable_conv2d('conv_ds_14', self.conv5_6,
                                                      width_multiplier=self.width_multiplier,
                                                      num_filters=1024, kernel_size=(3, 3), padding='SAME',
                                                      stride=(1, 1),
                                                      batchnorm_enabled=True, is_training=self.train_flag,
                                                      l2_strength=self.wd)

            self.score_fr = self.conv6_1

            self.feed1 = self.conv4_2

            self.feed2 = self.conv3_2

    def load(self):
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='mobilenet_encoder')

    print("\nEncoder MobileNet is built successfully\n\n")
