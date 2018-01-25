import tensorflow as tf


def max_pool_2d(x, size=(2, 2), stride=(2, 2), name='pooling'):
    """
    Max pooling 2D Wrapper
    :param x: (tf.tensor) The input to the layer (N,H,W,C).
    :param size: (tuple) This specifies the size of the filter as well as the stride.
    :param name: (string) Scope name.
    :return: The output is the same input but halfed in both width and height (N,H/2,W/2,C).
    """
    size_x, size_y = size
    stride_x, stride_y = stride
    return tf.nn.max_pool(x, ksize=[1, size_x, size_y, 1], strides=[1, stride_x, stride_y, 1], padding='VALID',
                          name=name)


def avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID'):
    """
        Average pooling 2D Wrapper
        :param x: (tf.tensor) The input to the layer (N,H,W,C).
        :param size: (tuple) This specifies the size of the filter as well as the stride.
        :param name: (string) Scope name.
        :return: The output is the same input but halfed in both width and height (N,H/2,W/2,C).
    """
    size_x, size_y = size
    stride_x, stride_y = stride
    return tf.nn.avg_pool(x, ksize=[1, size_x, size_y, 1], strides=[1, stride_x, stride_y, 1], padding=padding, name=name)

def upsample_2d(x, size=(2, 2), name='upsampling'):
    """
    Bilinear Upsampling 2D Wrapper
    :param x: (tf.tensor) The input to the layer (N,H,W,C).
    :param size: (tuple) This specifies the size of the filter as well as the stride.
    :param name: (string) Scope name
    :return: The output is the same input but doubled in both width and height (N,2H,2W,C).
    """
    h, w, _ = x.get_shape().as_list()[1:]
    size_x, size_y = size
    output_h = h * size_x
    output_w = w * size_y
    return tf.image.resize_bilinear(x, (output_h, output_w), align_corners=None, name=name)
