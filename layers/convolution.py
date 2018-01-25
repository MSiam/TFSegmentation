from layers.utils import *
from layers.pooling import max_pool_2d, avg_pool_2d
import tensorflow as tf
import pdb

def __conv2d_p(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
               initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    """
    Convolution 2D Wrapper
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param w: (tf.tensor) pretrained weights (if None, it means no pretrained weights)
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)
    :return out: The output of the layer. (N, H', W', num_filters)
    """
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], num_filters]

        with tf.name_scope('layer_weights'):
            if w == None:
                w = variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            variable_summaries(w)
        with tf.name_scope('layer_biases'):
            if isinstance(bias, float):
                bias = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(bias))
            variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.conv2d(x, w, stride, padding)
            out = tf.nn.bias_add(conv, bias)

    return out


def __atrous_conv2d_p(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', dilation_rate=1,
                      initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    """
    Atrous Convolution 2D Wrapper
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param w: (tf.tensor) pretrained weights
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param dilation_rate: (integer) The amount of dilation required. If equals 1, it means normal convolution.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)
    :return out: The output of the layer. (N, H', W', num_filters)
    """
    with tf.variable_scope(name):
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], num_filters]

        with tf.name_scope('layer_weights'):
            if w == None:
                w = variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            variable_summaries(w)
        with tf.name_scope('layer_biases'):
            if isinstance(bias, float):
                bias = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(bias))
            variable_summaries(bias)
        with tf.name_scope('layer_atrous_conv2d'):
            conv = tf.nn.atrous_conv2d(x, w, dilation_rate, padding)
            out = tf.nn.bias_add(conv, bias)

    return out


def __conv2d_transpose_p(name, x, w=None, output_shape=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                         l2_strength=0.0,
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
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)
    :return out: The output of the layer. (output_shape[0], output_shape[1], output_shape[2], output_shape[3])
    """
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], output_shape[-1], x.shape[-1]]
        if w == None:
            w = get_deconv_filter(kernel_shape, l2_strength)
        variable_summaries(w)
        deconv = tf.nn.conv2d_transpose(x, w, tf.stack(output_shape), strides=stride, padding=padding)
        if isinstance(bias, float):
            bias = tf.get_variable('layer_biases', [output_shape[-1]], initializer=tf.constant_initializer(bias))
        variable_summaries(bias)
        out = tf.nn.bias_add(deconv, bias)

    return out

def __depthwise_conv2d_atrous_p(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                         initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0,
                         dilation_factor= 1):
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], 1]

        with tf.name_scope('layer_weights'):
            if w == None:
                w = variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            variable_summaries(w)
        with tf.name_scope('layer_biases'):
            if isinstance(bias, float):
                bias = tf.get_variable('biases', [x.shape[-1]], initializer=tf.constant_initializer(bias))
            variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.depthwise_conv2d(x, w, stride, padding, [dilation_factor, dilation_factor])
            out = tf.nn.bias_add(conv, bias)

    return out


def __depthwise_conv2d_p(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                         initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], 1]

        with tf.name_scope('layer_weights'):
            if w == None:
                w = variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            variable_summaries(w)
        with tf.name_scope('layer_biases'):
            if isinstance(bias, float):
                bias = tf.get_variable('biases', [x.shape[-1]], initializer=tf.constant_initializer(bias))
            variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
#            pdb.set_trace()
            conv = tf.nn.depthwise_conv2d(x, w, stride, padding)
            out = tf.nn.bias_add(conv, bias)

    return out


def conv2d(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
           initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0,
           activation=None, batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=-1,
           is_training=True):
    """
    This block is responsible for a convolution 2D layer followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param max_pool_enabled:  (boolean) for enabling max-pooling 2x2 to decrease width and height by a factor of 2.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return: The output tensor of the layer (N, H', W', C').
    """
    with tf.variable_scope(name) as scope:
        conv_o_b = __conv2d_p(scope, x=x, w=w, num_filters=num_filters, kernel_size=kernel_size, stride=stride,
                              padding=padding,
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

        def dropout_with_keep():
            return tf.nn.dropout(conv_a, dropout_keep_prob)

        def dropout_no_keep():
            return tf.nn.dropout(conv_a, 1.0)

        if dropout_keep_prob != -1:
            conv_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            conv_o_dr = conv_a

        conv_o = conv_o_dr
        if max_pool_enabled:
            conv_o = max_pool_2d(conv_o_dr)

    return conv_o


def atrous_conv2d(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', dilation_rate=1,
                  initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0,
                  activation=None, batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=-1,
                  is_training=True):
    """
    This block is responsible for a Dilated convolution 2D layer followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param dilation_rate: (integer) The amount of dilation required. If equals 1, it means normal convolution.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param max_pool_enabled:  (boolean) for enabling max-pooling 2x2 to decrease width and height by a factor of 2.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return: The output tensor of the layer (N, H', W', C').
    """
    with tf.variable_scope(name) as scope:
        conv_o_b = __atrous_conv2d_p(scope, x=x, w=w, num_filters=num_filters, kernel_size=kernel_size,
                                     padding=padding, dilation_rate=dilation_rate,
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

        def dropout_with_keep():
            return tf.nn.dropout(conv_a, dropout_keep_prob)

        def dropout_no_keep():
            return tf.nn.dropout(conv_a, 1.0)

        if dropout_keep_prob != -1:
            conv_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            conv_o_dr = conv_a

        conv_o = conv_o_dr
        if max_pool_enabled:
            conv_o = max_pool_2d(conv_o_dr)

    return conv_o


def conv2d_transpose(name, x, w=None, output_shape=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                     l2_strength=0.0,
                     bias=0.0, activation=None, batchnorm_enabled=False, dropout_keep_prob=-1,
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
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return out: The output of the layer. (output_shape[0], output_shape[1], output_shape[2], output_shape[3])
    """
    with tf.variable_scope(name) as scope:
        conv_o_b = __conv2d_transpose_p(name=scope, x=x, w=w, output_shape=output_shape, kernel_size=kernel_size,
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

        def dropout_with_keep():
            return tf.nn.dropout(conv_a, dropout_keep_prob)

        def dropout_no_keep():
            return tf.nn.dropout(conv_a, 1.0)

        if dropout_keep_prob != -1:
            conv_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            conv_o_dr = conv_a

        conv_o = conv_o_dr

        return conv_o


def load_conv_layer(x, name, pretrained_weights, pooling=False, trainable=True, l2_strength=0.0, stride=(1, 1),
                    padding='SAME', dropout_keep_prob=-1, batchnorm_enabled=False, is_training=True):
    w = load_conv_filter(name, pretrained_weights, l2_strength, trainable=trainable)
    biases = load_bias(name, pretrained_weights, trainable=trainable)
    return conv2d(name, x=x, w=w, l2_strength=l2_strength, bias=biases, activation=tf.nn.relu,
                  max_pool_enabled=pooling, stride=stride, padding=padding, dropout_keep_prob=dropout_keep_prob,
                  is_training=is_training, batchnorm_enabled=batchnorm_enabled)


def load_depthwise_separable_conv_layer(x, name, pretrained_depthwise_weights, pretrained_pointwise_weights,
                                        trainable=True,
                                        l2_strength=0.0, width_multiplier=1.0, stride=(1, 1), padding='SAME',
                                        is_training=True):
    w_depthwise = load_conv_filter(name, pretrained_depthwise_weights, l2_strength, trainable=trainable)
    biases_depthwise = load_bias(name, pretrained_depthwise_weights, trainable=trainable)
    w_pointwise = load_conv_filter(name, pretrained_pointwise_weights, l2_strength, trainable=trainable)
    biases_pointwise = load_bias(name, pretrained_pointwise_weights, trainable=trainable)
    return depthwise_separable_conv2d(name, x=x, w_depthwise=w_depthwise,
                                      w_pointwise=w_pointwise, l2_strength=l2_strength,
                                      width_multiplier=width_multiplier,
                                      biases=(biases_depthwise, biases_pointwise),
                                      activation=tf.nn.relu, padding=padding, stride=stride, is_training=is_training)


def depthwise_conv2d(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),dilation_factor=1,
                     initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0, activation=None,
                     batchnorm_enabled=False, is_training=True):
    with tf.variable_scope(name) as scope:
        if dilation_factor>1:
             conv_o_b = __depthwise_conv2d_atrous_p(name=scope, x=x, w=w, kernel_size=kernel_size, padding=padding,
                                        stride=stride, initializer=initializer, l2_strength=l2_strength, bias=bias,
                                        dilation_factor= dilation_factor)
        else:
            conv_o_b = __depthwise_conv2d_p(name=scope, x=x, w=w, kernel_size=kernel_size, padding=padding,
                                        stride=stride, initializer=initializer, l2_strength=l2_strength, bias=bias)

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
    return conv_a


def depthwise_separable_conv2d(name, x, w_depthwise=None, w_pointwise=None, width_multiplier=1.0, num_filters=16,
                               kernel_size=(3, 3),
                               padding='SAME', stride=(1, 1),
                               initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, biases=(0.0, 0.0),
                               activation=None, batchnorm_enabled=True,
                               is_training=True):
    total_num_filters = int(round(num_filters * width_multiplier))
    with tf.variable_scope(name) as scope:
        conv_a = depthwise_conv2d('depthwise', x=x, w=w_depthwise, kernel_size=kernel_size, padding=padding,
                                  stride=stride,
                                  initializer=initializer, l2_strength=l2_strength, bias=biases[0],
                                  activation=activation,
                                  batchnorm_enabled=batchnorm_enabled, is_training=is_training)

        conv_o = conv2d('pointwise', x=conv_a, w=w_pointwise, num_filters=total_num_filters, kernel_size=(1, 1),
                        initializer=initializer, l2_strength=l2_strength, bias=biases[1], activation=activation,
                        batchnorm_enabled=batchnorm_enabled, is_training=is_training)

    return conv_o

def depthwise_separable_atrous_conv2d(name, x, w_depthwise=None, w_pointwise=None, width_multiplier=1.0, num_filters=16,
                               kernel_size=(3, 3), dilation_factor=1,
                               padding='SAME', stride=(1, 1),
                               initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, biases=(0.0, 0.0),
                               activation=None, batchnorm_enabled=True,
                               is_training=True):
    total_num_filters = int(round(num_filters * width_multiplier))
    with tf.variable_scope(name) as scope:
        conv_a = depthwise_conv2d('depthwise', x=x, w=w_depthwise, kernel_size=kernel_size, padding=padding,
                                  stride=stride, dilation_factor=filation_factor,
                                  initializer=initializer, l2_strength=l2_strength, bias=biases[0],
                                  activation=activation,
                                  batchnorm_enabled=batchnorm_enabled, is_training=is_training)

        conv_o = conv2d('pointwise', x=conv_a, w=w_pointwise, num_filters=total_num_filters, kernel_size=(1, 1),
                        initializer=initializer, l2_strength=l2_strength, bias=biases[1], activation=activation,
                        batchnorm_enabled=batchnorm_enabled, is_training=is_training)

    return conv_o


# ShuffleNet layer methods


############################################################################################################
# ShuffleNet unit methods

def grouped_conv2d(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                   initializer=tf.contrib.layers.xavier_initializer(), num_groups=1, l2_strength=0.0, bias=0.0,
                   activation=None, batchnorm_enabled=False, dropout_keep_prob=-1,
                   is_training=True):
    with tf.variable_scope(name) as scope:
        sz = x.get_shape()[3].value // num_groups
        conv_side_layers = [
            conv2d(name + "_" + str(i), x[:, :, :, i * sz:i * sz + sz], w, num_filters // num_groups, kernel_size,
                   padding,
                   stride,
                   initializer,
                   l2_strength, bias, activation=None,
                   batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=dropout_keep_prob,
                   is_training=is_training) for i in
            range(num_groups)]
        conv_g = tf.concat(conv_side_layers, axis=-1)

        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_g, training=is_training, epsilon=1e-5)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_g
            else:
                conv_a = activation(conv_g)

        return conv_a


def shufflenet_unit(name, x, w=None, num_groups=1, group_conv_bottleneck=True, num_filters=16, stride=(1, 1),
                    l2_strength=0.0, bias=0.0, batchnorm_enabled=True, is_training=True, fusion='add', dilation=1):
    # Paper parameters. If you want to change them feel free to pass them as method parameters.
    activation = tf.nn.relu

    with tf.variable_scope(name) as scope:
        residual = x
        bottleneck_filters = (num_filters // 4) if fusion == 'add' else (num_filters - residual.get_shape()[
            3].value) // 4

        if group_conv_bottleneck:
            bottleneck = grouped_conv2d('Gbottleneck', x=x, w=None, num_filters=bottleneck_filters, kernel_size=(1, 1),
                                        padding='VALID',
                                        num_groups=num_groups, l2_strength=l2_strength, bias=bias,
                                        activation=activation,
                                        batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            #_debug(bottleneck)
            shuffled = channel_shuffle('channel_shuffle', bottleneck, num_groups)
            #_debug(shuffled)
        else:
            bottleneck = conv2d('bottleneck', x=x, w=None, num_filters=bottleneck_filters, kernel_size=(1, 1),
                                padding='VALID', l2_strength=l2_strength, bias=bias, activation=activation,
                                batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            #_debug(bottleneck)
            shuffled = bottleneck

        if dilation>1:
            depthwise = depthwise_conv2d('depthwise', x=shuffled, w=None, stride=stride, l2_strength=l2_strength,
                                     padding='SAME', bias=bias, dilation_factor= dilation,
                                     activation=None, batchnorm_enabled=batchnorm_enabled, is_training=is_training)
        else:
            padded = tf.pad(shuffled, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
            depthwise = depthwise_conv2d('depthwise', x=padded, w=None, stride=stride, l2_strength=l2_strength,
                                     padding='VALID', bias=bias, dilation_factor= dilation,
                                     activation=None, batchnorm_enabled=batchnorm_enabled, is_training=is_training)

        #_debug(depthwise)
        if stride == (2, 2):
            residual_pooled = avg_pool_2d(residual, size=(3, 3), stride=stride, padding='SAME')
            #_debug(residual_pooled)
        else:
            residual_pooled = residual
            #_debug(residual_pooled)
        if fusion == 'concat':
            group_conv1x1 = grouped_conv2d('Gconv1x1', x=depthwise, w=None,
                                           num_filters=num_filters - residual.get_shape()[3].value,
                                           kernel_size=(1, 1),
                                           padding='VALID',
                                           num_groups=num_groups, l2_strength=l2_strength, bias=bias,
                                           activation=None,
                                           batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            return activation(tf.concat([residual_pooled, group_conv1x1], axis=-1))
        elif fusion == 'add':
            group_conv1x1 = grouped_conv2d('Gconv1x1', x=depthwise, w=None,
                                           num_filters=num_filters,
                                           kernel_size=(1, 1),
                                           padding='VALID',
                                           num_groups=num_groups, l2_strength=l2_strength, bias=bias,
                                           activation=None,
                                           batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            #_debug(group_conv1x1)
            residual_match = residual_pooled
            # This is used if the number of filters of the residual block is different from that
            # of the group convolution.
            if num_filters != residual_pooled.get_shape()[3].value:
                residual_match = conv2d('residual_match', x=residual_pooled, w=None, num_filters=num_filters,
                                        kernel_size=(1, 1),
                                        padding='VALID', l2_strength=l2_strength, bias=bias, activation=None,
                                        batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            return activation(group_conv1x1 + residual_match)
        else:
            raise ValueError("Specify whether the fusion is \'concat\' or \'add\'")


def channel_shuffle(name, x, num_groups):
    with tf.variable_scope(name) as scope:
        n, h, w, c = x.shape.as_list()
        x_reshaped = tf.reshape(x, [-1, h, w, num_groups, c // num_groups])
        x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
        output = tf.reshape(x_transposed, [-1, h, w, c])
    return output
