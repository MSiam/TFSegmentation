import tensorflow as tf
from layers.convolution import conv2d
from layers.pooling import max_pool_2d_2x2


def conv2d_full(name, x, num_filters, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0,
                activation='relu', batchnorm_enabled=False, max_pool_enabled=True, dropout_keep_prob=1.0,
                is_training=True):
    """
    This block is responsible for a conv2d layer followed by optional (non-linearity, dropout, max-pooling).
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
    :param activation: (string) The activation function applied after the convolution operation. At this moment,'relu' and 'linear' are supported.
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

            conv_a = conv_o_bn
            if activation == 'relu':
                conv_a = tf.nn.relu(conv_o_bn)
            elif activation == 'linear':
                pass
        else:
            conv_a = conv_o_b
            if activation == 'relu':
                conv_a = tf.nn.relu(conv_o_b)
            elif activation == 'linear':
                pass

        conv_o_dr = tf.nn.dropout(conv_a, dropout_keep_prob)

        conv_o = conv_o_dr
        if max_pool_enabled:
            conv_o = max_pool_2d_2x2(conv_o_dr)

    return conv_o
