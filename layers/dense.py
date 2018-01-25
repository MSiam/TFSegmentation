from layers.utils import *
from layers.convolution import conv2d
import tensorflow as tf


def dense_p(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
            bias=0.0):
    """
    Fully connected layer
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, D).
    :param output_dim: (integer) It specifies H, the output second dimension of the fully connected layer [ie:(N, H)]
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)
    :return out: The output of the layer. (N, H)
    """
    n_in = x.get_shape()[-1].value
    with tf.variable_scope(name):
        if w == None:
            w = variable_with_weight_decay([n_in, output_dim], initializer, l2_strength)
        variable_summaries(w)
        if isinstance(bias, float):
            bias = tf.get_variable("layer_biases", [output_dim], tf.float32, tf.constant_initializer(bias))
        variable_summaries(bias)
        output = tf.nn.bias_add(tf.matmul(x, w), bias)
        return output


def dense(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
          bias=0.0,
          activation=None, batchnorm_enabled=False, dropout_keep_prob=-1,
          is_training=True
          ):
    """
    This block is responsible for a fully connected followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, D).
    :param output_dim: (integer) It specifies H, the output second dimension of the fully connected layer [ie:(N, H)]
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout) 
    :return out: The output of the layer. (N, H)
    """
    with tf.variable_scope(name) as scope:
        dense_o_b = dense_p(name=scope, x=x, w=w, output_dim=output_dim, initializer=initializer,
                            l2_strength=l2_strength,
                            bias=bias)

        if batchnorm_enabled:
            dense_o_bn = tf.layers.batch_normalization(dense_o_b, training=is_training)
            if not activation:
                dense_a = dense_o_bn
            else:
                dense_a = activation(dense_o_bn)
        else:
            if not activation:
                dense_a = dense_o_b
            else:
                dense_a = activation(dense_o_b)

        def dropout_with_keep():
            return tf.nn.dropout(dense_a, dropout_keep_prob)

        def dropout_no_keep():
            return tf.nn.dropout(dense_a, 1.0)

        if dropout_keep_prob != -1:
            dense_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            dense_o_dr = dense_a

        dense_o = dense_o_dr
    return dense_o


def flatten(x):
    """
    Flatten a (N,H,W,C) input into (N,D) output. Used for fully connected layers after conolution layers
    :param x: (tf.tensor) representing input
    :return: flattened output
    """
    all_dims_exc_first = np.prod([v.value for v in x.get_shape()[1:]])
    o = tf.reshape(x, [-1, all_dims_exc_first])
    return o


def load_dense_layer(reduced_flag, x, name, pretrained_weights, num_classes=20, activation=None, dropout=-1,
                     train=False,
                     trainable=True, l2_strength=0.0):
    """
    Load fully connected layers from pretrained weights in case of full vgg
    in case of reduced vgg initialize randomly
    """
    if not reduced_flag:
        if name == 'fc6':
            w = get_dense_weight_reshape(name, pretrained_weights, [7, 7, 512, 4096], trainable=trainable)
        elif name == 'score_fr':
            name = 'fc8'
            w = get_dense_weight_reshape(name, pretrained_weights, [1, 1, 4096, 1000], num_classes=num_classes,
                                         trainable=trainable)
        else:
            w = get_dense_weight_reshape(name, pretrained_weights, [1, 1, 4096, 4096], trainable=trainable)

        biases = load_bias(name, pretrained_weights, num_classes=num_classes, trainable=trainable)
        return conv2d(name, x=x, w=w, l2_strength=l2_strength, bias=biases,
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

        return conv2d(name, x=x, num_filters=num_channels, kernel_size=kernel_size, l2_strength=l2_strength,
                      activation=activation, dropout_keep_prob=dropout, is_training=train)
