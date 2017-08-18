from layers.utils import *
from layers.pooling import max_pool_2d
import tensorflow as tf


def conv2d_pre(name, x, w, padding='SAME', stride=(1, 1), bias=0.0):
    """
    Convolution 2D Wrapper
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param w: pretrained weights
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param bias: (float) Amount of bias.
    :return out: The output of the layer. (N, H', W', num_filters)
    """
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]

        with tf.name_scope('layer_weights'):
            variable_summaries(w)
        with tf.name_scope('layer_biases'):
            variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.conv2d(x, w, stride, padding)
            out = tf.nn.bias_add(conv, bias)
    return out


def conv2d(name, x, num_filters, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
           initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    """
    Convolution 2D Wrapper
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :return out: The output of the layer. (N, H', W', num_filters)
    """
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], num_filters]

        with tf.name_scope('layer_weights'):
            w = variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            variable_summaries(w)
        with tf.name_scope('layer_biases'):
            b = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(bias))
            variable_summaries(b)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.conv2d(x, w, stride, padding)
            out = tf.nn.bias_add(conv, b)

    return out


def atrous_conv2d(name, x, num_filters, kernel_size=(3, 3), padding='SAME', dilation_rate=1,
                  initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    """
    Atrous Convolution 2D Wrapper
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param dilation_rate: (integer) The amount of dilation required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :return out: The output of the layer. (N, H', W', num_filters)
    """
    with tf.variable_scope(name):
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], num_filters]

        with tf.name_scope('layer_weights'):
            w = variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            variable_summaries(w)
        with tf.name_scope('layer_biases'):
            b = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(bias))
            variable_summaries(b)
        with tf.name_scope('layer_atrous_conv2d'):
            conv = tf.nn.atrous_conv2d(x, w, dilation_rate, padding)
            out = tf.nn.bias_add(conv, b)

    return out


def conv2d_transpose(name, x, output_shape, kernel_size=(3, 3), padding='SAME', stride=(1, 1), l2_strength=0.0,
                     bias=0.0):
    """
    Convolution Transpose 2D Wrapper
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param output_shape: (Array) [N, H', W', C'] The shape of the corresponding output.
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :return out: The output of the layer. (output_shape[0], output_shape[1], output_shape[2], output_shape[3])
    """
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], output_shape[-1], x.shape[-1]]

        w = get_deconv_filter(kernel_shape, l2_strength)
        deconv = tf.nn.conv2d_transpose(x, w, tf.stack(output_shape), strides=stride, padding=padding)

        b = tf.get_variable('layer_biases', [output_shape[-1]], initializer=tf.constant_initializer(bias))
        out = tf.nn.bias_add(deconv, b)

    return out


def conv2d_f_pre(name, x, w, padding='SAME', stride=(1, 1),
                 l2_strength=0.0, bias=0.0,
                 activation=None, batchnorm_enabled=False, max_pool_enabled=True, dropout_keep_prob=1.0,
                 is_training=True):
    """
    This block is responsible for a convolution 2D layer followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param w: pretrained weights
    :param padding: (string) The amount of padding required.s
    :param stride: (integer tuple) The stride required.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param max_pool_enabled:  (boolean) for enabling max-pooling 2x2 to decrease width and height by a factor of 2.
    :param dropout_keep_prob: (float) for the probability of keeping neurons.
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return: The output tensor of the layer (N, H', W', C').
    """
    with tf.name_scope(name) as scope:
        conv_o_b = conv2d_pre(scope, x, w, stride=stride, padding=padding, bias=bias)

        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)

        conv_o_dr = tf.nn.dropout(conv_a, dropout_keep_prob)

        conv_o = conv_o_dr
        if max_pool_enabled:
            conv_o = max_pool_2d(scope, conv_o_dr)

    return conv_o


def conv2d_f(name, x, num_filters, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
             initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0,
             activation=None, batchnorm_enabled=False, max_pool_enabled=True, dropout_keep_prob=1.0,
             is_training=True):
    """
    This block is responsible for a convolution 2D layer followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.s
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param max_pool_enabled:  (boolean) for enabling max-pooling 2x2 to decrease width and height by a factor of 2.
    :param dropout_keep_prob: (float) for the probability of keeping neurons.
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return: The output tensor of the layer (N, H', W', C').
    """
    with tf.name_scope(name) as scope:
        conv_o_b = conv2d(scope, x, num_filters, kernel_size=kernel_size, stride=stride, padding=padding,
                          initializer=initializer, l2_strength=l2_strength, bias=bias)

        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)

        conv_o_dr = tf.nn.dropout(conv_a, dropout_keep_prob)

        conv_o = conv_o_dr
        if max_pool_enabled:
            conv_o = max_pool_2d(scope, conv_o_dr)

    return conv_o


def deconv2d_f(name, x, output_shape, kernel_size=(3, 3), padding='SAME', stride=(1, 1), l2_strength=0.0,
               bias=0.0, activation=None, batchnorm_enabled=False, max_pool_enabled=True, dropout_keep_prob=1.0,
               is_training=True):
    """
    This block is responsible for a convolution transpose 2D followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param output_shape: (Array) [N, H', W', C'] The shape of the corresponding output.
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param max_pool_enabled:  (boolean) for enabling max-pooling 2x2 to decrease width and height by a factor of 2.
    :param dropout_keep_prob: (float) for the probability of keeping neurons.
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return out: The output of the layer. (output_shape[0], output_shape[1], output_shape[2], output_shape[3])
    """
    with tf.name_scope(name) as scope:
        conv_o_b = conv2d_transpose(name=scope, x=x, output_shape=output_shape, kernel_size=kernel_size,
                                    padding=padding, stride=stride,
                                    l2_strength=l2_strength,
                                    bias=bias)

        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)

        conv_o_dr = tf.nn.dropout(conv_a, dropout_keep_prob)

        conv_o = conv_o_dr
        if max_pool_enabled:
            conv_o = max_pool_2d(scope, conv_o_dr)

        return conv_o
