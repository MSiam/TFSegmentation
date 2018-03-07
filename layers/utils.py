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
        weight_decay = tf.multiply(tf.nn.l2_loss(w), wd, name='w_loss')
        tf.add_to_collection(collection_name, weight_decay)
    variable_summaries(w)
    return w

def variable_with_weight_decay2(kernel_shape, initializer, wd, trainable=True):
    """
    Create a variable with L2 Regularization (Weight Decay)
    :param kernel_shape: the size of the convolving weight kernel.
    :param initializer: The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param wd:(weight decay) L2 regularization parameter.
    :return: The weights of the kernel initialized. The L2 loss is added to the loss collection.
    """
    w = tf.get_variable('kernel', kernel_shape, tf.float32, initializer=initializer, trainable=trainable)

    if trainable:
        collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.multiply(tf.nn.l2_loss(w), wd, name='w_loss')
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


def load_conv_filter(name, pretrained_weights, l2_strength=0.0, trainable=True):
    with tf.variable_scope(name):
        init = tf.constant_initializer(value=pretrained_weights[name][0], dtype=tf.float32)
        shape = pretrained_weights[name][0].shape
        print('Layer name: %s' % name)
        print('Layer shape: %s' % str(shape))
        var = tf.get_variable(name="filters", initializer=init, shape=shape, trainable=trainable)
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), l2_strength, name='weight_loss')
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_decay)
        return var


def load_bias(name, pretrained_weights, trainable=True, num_classes=None):
    with tf.variable_scope(name):
        bias_weights = pretrained_weights[name][1]
        shape = pretrained_weights[name][1].shape
        if name == 'fc8':
            bias_weights = _bias_reshape(bias_weights, shape[0], num_classes)
            shape = [num_classes]
        init = tf.constant_initializer(value=bias_weights, dtype=tf.float32)
        var = tf.get_variable(name="biases", initializer=init, shape=shape, trainable=trainable)
        return var


def get_dense_weight_reshape(name, pretrained_weights, shape, trainable=True, num_classes=None):
    with tf.variable_scope(name):
        print('Layer name: %s' % name)
        print('Layer shape: %s' % shape)
        weights = pretrained_weights[name][0]
        weights = weights.reshape(shape)
        if num_classes is not None:
            weights = _summary_reshape(weights, shape, num_new=num_classes)
        init = tf.constant_initializer(value=weights, dtype=tf.float32)
        var = tf.get_variable(name="weights", initializer=init, shape=shape, trainable=trainable)
        return var


def _bias_reshape(bweight, num_orig, num_new):
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


def _summary_reshape(fweight, shape, num_new):
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
