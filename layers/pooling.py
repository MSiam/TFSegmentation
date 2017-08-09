import tensorflow as tf


def max_pool_2d_2x2(x):
    """
    Max pooling 2x2 Wrapper
    :param x: (tf.tensor) The input to the layer (N,H,W,C).
    :return: The output is the same input but halfed in both width and height (N,H/2,W/2,C). 
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
