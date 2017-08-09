from layers.utils import *


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
        kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], num_filters]

        with tf.name_scope('weights'):
            w = variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            variable_summaries(w)
        with tf.name_scope('biases'):
            b = tf.get_variable('biases_conv', [num_filters], initializer=tf.constant_initializer(bias))
            variable_summaries(b)
        with tf.name_scope('conv2d'):
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
        kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], num_filters]

        with tf.name_scope('weights'):
            w = variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            variable_summaries(w)
        with tf.name_scope('biases'):
            b = tf.get_variable('biases_conv', [num_filters], initializer=tf.constant_initializer(bias))
            variable_summaries(b)
        with tf.name_scope('atrous_conv2d'):
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
    :return:
    """
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], output_shape[-1], x.get_shape()[-1]]

        w = get_deconv_filter(kernel_shape, l2_strength)
        deconv = tf.nn.conv2d_transpose(x, w, tf.stack(output_shape), strides=stride, padding=padding)

        b = tf.get_variable('biases_deconv', [output_shape[-1]], initializer=tf.constant_initializer(bias))
        out = tf.nn.bias_add(deconv, b)

    return out
