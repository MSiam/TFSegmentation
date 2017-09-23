import numpy
import tensorflow as tf


def load_webp_from_bytes(data):
  # needs sudo apt-get install python-webm
  from webm.decode import DecodeRGB
  x = DecodeRGB(data)
  res = numpy.array(x.bitmap).reshape((x.height, x.width, 3))
  return res


def load_webp_from_file(fn):
  with open(fn) as f:
    data = f.read()
  return load_webp_from_bytes(data)


def decode_webp_tensorflow(fn):
  return tf.py_func(load_webp_from_bytes, [fn], tf.uint8)
