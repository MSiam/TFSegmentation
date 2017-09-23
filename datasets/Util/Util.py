import os
import numpy as np
import tensorflow as tf
from skimage import color

from datasets.Util.flo_Reader import read_flo_file
from datasets.Util.python_pfm import readPFM


def unique_list(l):
  res = []
  for x in l:
    if x not in res:
      res.append(x)
  return res


def create_index_image(height, width):
  y = tf.range(height)
  x = tf.range(width)
  grid = tf.meshgrid(x, y)
  index_img = tf.stack((grid[1], grid[0]), axis=2)
  return index_img


def smart_shape(x):
  shape = x.get_shape().as_list()
  tf_shape = tf.shape(x)
  for i, s in enumerate(shape):
    if s is None:
      shape[i] = tf_shape[i]
  return shape


def read_pfm(fn):
  return readPFM(fn)[0]


def username():
  return os.environ["USER"]


def _postprocess_flow(x, flow_as_angle):
  if flow_as_angle:
    assert False, "not implemented yet"
  else:
    # divide by 20 to get to a more useful range
    x /= 20.0
  return x


def load_flow_from_pfm(fn, flow_as_angle=False):
  # 3rd channel is all zeros
  flow = read_pfm(fn)[:, :, :2]
  flow = _postprocess_flow(flow, flow_as_angle)
  return flow


def load_flow_from_flo(fn, flow_as_angle):
  flow = read_flo_file(fn)
  flow = _postprocess_flow(flow, flow_as_angle)
  return flow

def get_masked_image(img, mask, multiplier=0.6):
  """

  :param img: The image to be masked.
  :param mask: Binary mask to be applied. The object should be represented by 1 and the background by 0
  :param multiplier: Floating point multiplier that decides the colour of the mask.
  :return: Masked image
  """
  img_mask = np.zeros_like(img)
  indices = np.where(mask == 1)
  img_mask[indices[0], indices[1], 1] = 1
  img_mask_hsv = color.rgb2hsv(img_mask)
  img_hsv = color.rgb2hsv(img)
  img_hsv[indices[0], indices[1], 0] = img_mask_hsv[indices[0], indices[1], 0]
  img_hsv[indices[0], indices[1], 1] = img_mask_hsv[indices[0], indices[1], 1] * multiplier

  return color.hsv2rgb(img_hsv)
