from layers.convolution import conv2d
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

        print("\nEncoder VGG is built successfully\n\n")

    def load_fc_layer(self, bottom, name, num_classes=20, activation=None, dropout=1.0, train=False, trainable=True):
        """
        Load fully connected layers from pretrained weights in case of full vgg
        in case of reduced vgg initialize randomly
        """
        if not self.reduced_flag:
            if name == 'fc6':
                w = self.get_fc_weight_reshape(name, [7, 7, 512, 4096], trainable=trainable)
            elif name == 'score_fr':
                name = 'fc8'
                w = self.get_fc_weight_reshape(name, [1, 1, 4096, 1000], num_classes=num_classes, trainable=trainable)
            else:
                w = self.get_fc_weight_reshape(name, [1, 1, 4096, 4096], trainable=trainable)

            biases = self.get_bias(name, num_classes=num_classes, trainable=trainable)
            return conv2d(name, x=bottom, w=w, l2_strength=self.wd, bias=biases,
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

            return conv2d(name, x=bottom, num_filters=num_channels, kernel_size=kernel_size, l2_strength=self.wd, activation=activation, dropout_keep_prob=dropout, is_training=train)

    def load_conv_layer(self, bottom, name, pooling=False, trainable=True):
        w = self.get_conv_filter(name, trainable=trainable)
        biases = self.get_bias(name, trainable=trainable)
        return conv2d(name, x=bottom, w=w, l2_strength=self.wd, bias=biases, activation=tf.nn.relu, max_pool_enabled=pooling)

    """
    ==============================================================
    Next Functions are helpers for loading pretrained weights
    ==============================================================
    """

    def get_conv_filter(self, name, trainable=True):
        with tf.variable_scope(name):
            init = tf.constant_initializer(value=self.pretrained_weights[name][0], dtype=tf.float32)
            shape = self.pretrained_weights[name][0].shape
            print('Layer name: %s' % name)
            print('Layer shape: %s' % str(shape))
            var = tf.get_variable(name="filters", initializer=init, shape=shape, trainable=trainable)
            if not tf.get_variable_scope().reuse:
                weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd, name='weight_loss')
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_decay)
            return var

    def get_bias(self, name, trainable=True, num_classes=None):
        with tf.variable_scope(name):
            bias_weights = self.pretrained_weights[name][1]
            shape = self.pretrained_weights[name][1].shape
            if name == 'fc8':
                bias_weights = self._bias_reshape(bias_weights, shape[0], num_classes)
                shape = [num_classes]
            init = tf.constant_initializer(value=bias_weights, dtype=tf.float32)
            var = tf.get_variable(name="biases", initializer=init, shape=shape, trainable=trainable)
            return var

    def get_fc_weight_reshape(self, name, shape, trainable=True, num_classes=None):
        with tf.variable_scope(name):
            print('Layer name: %s' % name)
            print('Layer shape: %s' % shape)
            weights = self.pretrained_weights[name][0]
            weights = weights.reshape(shape)
            if num_classes is not None:
                weights = self._summary_reshape(weights, shape, num_new=num_classes)
            init = tf.constant_initializer(value=weights, dtype=tf.float32)
            var = tf.get_variable(name="weights", initializer=init, shape=shape, trainable=trainable)
            return var

    def _bias_reshape(self, bweight, num_orig, num_new):
        """
        Build bias weights for filter produces with `_summary_reshape`
        """
        n_averaged_elements = num_orig // num_new
        avg_bweight = np.zeros(num_new)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx // n_averaged_elements
            if avg_idx == num_new:
                break
            avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
        return avg_bweight

    def _summary_reshape(self, fweight, shape, num_new):
        """
        Produce weights for a reduced fully-connected layer.

        FC8 of VGG produces 1000 classes. Most semantic segmentation
        task require much less classes. This reshapes the original weights
        to be used in a fully-convolutional layer which produces num_new
        classes. To archive this the average (mean) of n adjanced classes is
        taken.

        Consider reordering fweight, to perserve semantic meaning of the
        weights.

        Args:
          fweight: original weights
          shape: shape of the desired fully-convolutional layer
          num_new: number of new classes


        Returns:
          Filter weights for `num_new` classes.
        """
        num_orig = shape[3]
        shape[3] = num_new
        assert (num_new < num_orig)
        n_averaged_elements = num_orig // num_new
        avg_fweight = np.zeros(shape)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx // n_averaged_elements
            if avg_idx == num_new:
                break
            avg_fweight[:, :, :, avg_idx] = np.mean(
                fweight[:, :, :, start_idx:end_idx], axis=3)
        return avg_fweight
