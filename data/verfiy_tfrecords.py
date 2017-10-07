import tensorflow as tf
import numpy as np

# Reset the graph
tf.reset_default_graph()

filename = 'cscapesval.tfrecords'


def parser(record):
    keys_to_features = {
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'mask_raw': tf.FixedLenFeature([], tf.string)
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    image = tf.cast(tf.decode_raw(parsed['image_raw'], tf.uint8), tf.float32)
    annotation = tf.cast(tf.decode_raw(parsed['mask_raw'], tf.uint8), tf.int32)

    height = tf.cast(parsed['height'], tf.int32)
    width = tf.cast(parsed['width'], tf.int32)

    image_shape = tf.stack([height, width, 3])

    annotation_shape = tf.stack([height, width])

    image = tf.reshape(image, image_shape)
    annotation = tf.reshape(annotation, annotation_shape)

    return image, annotation


# Use `Dataset.map()` to build a pair of a feature dictionary and a label
# tensor for each example.
dataset = tf.contrib.data.TFRecordDataset(filename)
dataset = dataset.map(parser)
dataset = dataset.shuffle(buffer_size=500)
dataset = dataset.batch(5)

iterator = dataset.make_one_shot_iterator()

next_img = iterator.get_next()
x_pl, y_pl = next_img

with tf.Session() as sess:
    next_element = sess.run(next_img)
    x, y = next_element
    print(x)
    print(y)
    print(x.dtype)
    print(y.dtype)
    print(x.shape)
    print(y.shape)
    exit(0)
