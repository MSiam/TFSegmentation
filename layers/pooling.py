import tensorflow as tf


def max_pool_2d(name, x, size=(2, 2)):
    """
    Max pooling 2D Wrapper
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N,H,W,C).
    :param size: (tuple) This specifies the size of the filter as well as the stride.
    :return: The output is the same input but halfed in both width and height (N,H/2,W/2,C). 
    """
    size_x, size_y = size
    return tf.nn.max_pool(x, ksize=[1, size_x, size_y, 1], strides=[1, size_x, size_y, 1], padding='VALID',
                          name=name)


def upsample_2d(name, x, size=(2, 2)):
    """
    Bilinear Upsampling 2D Wrapper
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N,H,W,C).
    :param size: (tuple) This specifies the size of the filter as well as the stride.
    :return: The output is the same input but doubled in both width and height (N,2H,2W,C). 
    """
    H, W, _ = x.get_shape().as_list()[1:]
    size_x, size_y = size
    output_H = H * size_x
    output_W = W * size_y
    return tf.image.resize_bilinear(x, (output_H, output_W), align_corners=None, name=name)
