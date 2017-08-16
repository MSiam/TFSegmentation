from layers.convolution import conv2d_f_pre, conv2d_f
import numpy as np
import tensorflow as tf


# TODO Revise this class with menna

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

        # Convert RGB to BGR
        with tf.name_scope('Pre_Processing'):
            red, green, blue = tf.split(self.x_input, num_or_size_splits=3, axis=3)
            preprocessed_input = tf.concat([
                blue - VGG16.VGG_MEAN[0],
                green - VGG16.VGG_MEAN[1],
                red - VGG16.VGG_MEAN[2],
            ], 3)

        self.conv1_1 = self.load_conv_layer(preprocessed_input, 'conv1_1')
        self.conv1_2 = self.load_conv_layer(self.conv1_1, 'conv1_2', pooling=True)

        self.conv2_1 = self.load_conv_layer(self.conv1_2, 'conv2_1')
        self.conv2_2 = self.load_conv_layer(self.conv2_1, 'conv2_2', pooling=True)

        self.conv3_1 = self.load_conv_layer(self.conv2_2, 'conv3_1')
        self.conv3_2 = self.load_conv_layer(self.conv3_1, 'conv3_2')
        self.conv3_3 = self.load_conv_layer(self.conv3_2, 'conv3_3', pooling=True)

        self.conv4_1 = self.load_conv_layer(self.conv3_3, 'conv4_1')
        self.conv4_2 = self.load_conv_layer(self.conv4_1, 'conv4_2')
        self.conv4_3 = self.load_conv_layer(self.conv4_2, 'conv4_3', pooling=True)

        self.conv5_1 = self.load_conv_layer(self.conv4_3, 'conv5_1')
        self.conv5_2 = self.load_conv_layer(self.conv5_1, 'conv5_2')
        self.conv5_3 = self.load_conv_layer(self.conv5_2, 'conv5_3', pooling=True)

        self.fc6 = self.load_fc_layer(self.conv5_3, 'fc6', activation=tf.nn.relu, dropout=0.5, train=self.train_flag)
        self.fc7 = self.load_fc_layer(self.fc6, 'fc7', activation=tf.nn.relu, dropout=0.5, train=self.train_flag)
        self.score_fr = self.load_fc_layer(self.fc7, 'score_fr', num_classes=self.num_classes)

        self.feed1 = self.conv4_3
        self.feed2 = self.conv3_3

    def load_fc_layer(self, bottom, name, num_classes=20, activation=None, dropout=1.0, train=False):
        """
        Load fully connected layers from pretrained weights in case of full vgg
        in case of reduced vgg initialize randomly
        """
        if not self.reduced_flag:
            if name == 'fc6':
                w = self.get_fc_weight_reshape(name, [7, 7, 512, 4096])
            elif name == 'score_fr':
                name = 'fc8'
                w = self.get_fc_weight_reshape(name, [1, 1, 4096, 1000], num_classes=num_classes)
            else:
                w = self.get_fc_weight_reshape(name, [1, 1, 4096, 4096])

            biases = self.get_bias(name, num_classes=num_classes)
            return conv2d_f_pre(name, bottom, w, l2_strength=self.wd, bias=biases,
                                activation=activation, dropout_keep_prob=dropout, is_training=train)
        else:
            if name == 'fc6':
                num_channels = 512
                kernel_size = (7, 7)
            elif name == 'score_fr':
                name = 'fc8'
                num_channels = num_classes
                kernel_size = (1, 1)
            else:
                num_channels = 512
                kernel_size = (1, 1)

            return conv2d_f(name, bottom, num_channels, kernel_size=kernel_size, l2_strength=self.wd, activation=activation, dropout_keep_prob=dropout, is_training=train)

    def load_conv_layer(self, bottom, name, pooling=False):
        w = self.get_conv_filter(name)
        biases = self.get_bias(name)
        return conv2d_f_pre(name, bottom, w, l2_strength=self.wd, bias=biases, activation=tf.nn.relu, max_pool_enabled=pooling)

    """
    ==============================================================
    Next Functions are helpers for loading pretrained weights
    ==============================================================
    """

    def get_conv_filter(self, name):
        init = tf.constant_initializer(value=self.pretrained_weights[name][0], dtype=tf.float32)
        shape = self.pretrained_weights[name][0].shape
        print('Layer name: %s' % name)
        print('Layer shape: %s' % str(shape))
        var = tf.get_variable(name="filter", initializer=init, shape=shape)
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd, name='weight_loss')
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_decay)
        return var

    def get_bias(self, name, num_classes=None):
        bias_weights = self.pretrained_weights[name][1]
        shape = self.pretrained_weights[name][1].shape
        if name == 'fc8':
            bias_weights = self._bias_reshape(bias_weights, shape[0], num_classes)
            shape = [num_classes]
        init = tf.constant_initializer(value=bias_weights, dtype=tf.float32)
        var = tf.get_variable(name="biases", initializer=init, shape=shape)
        return var

    def get_fc_weight_reshape(self, name, shape, num_classes=None):
        print('Layer name: %s' % name)
        print('Layer shape: %s' % shape)
        weights = self.pretrained_weights[name][0]
        weights = weights.reshape(shape)
        if num_classes is not None:
            weights = self._summary_reshape(weights, shape, num_new=num_classes)
        init = tf.constant_initializer(value=weights, dtype=tf.float32)
        var = tf.get_variable(name="weights", initializer=init, shape=shape)
        return var
