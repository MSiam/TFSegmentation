import tensorflow as tf
import math
import numpy as np


def variable_with_weight_decay(kernel_shape, initializer, wd):
    """
    Create a variable with L2 Regularization (Weight Decay)
    :param kernel_shape: the size of the convolving weight kernel.
    :param initializer: The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param wd:(weight decay) L2 regularization parameter.
    :return: The weights of the kernel initialized. The L2 loss is added to the loss collection.
    """
    w = tf.get_variable('weights', kernel_shape, tf.float32, initializer=initializer)

    collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(
            tf.nn.l2_loss(w), wd, name='w_loss')
        tf.add_to_collection(collection_name, weight_decay)
    variable_summaries(w)
    return w


# Summaries for variables
def variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    :param var: variable to be summarized
    :return: None
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def get_deconv_filter(f_shape, l2_strength):
    """
    The initializer for the bilinear convolution transpose filters
    :param f_shape: The shape of the filter used in convolution transpose.
    :param l2_strength: L2 regularization parameter.
    :return weights: The initialized weights.
    """
    width = f_shape[0]
    height = f_shape[0]
    f = math.ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(height):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights, dtype=tf.float32)
    return variable_with_weight_decay(weights.shape, init, l2_strength)
