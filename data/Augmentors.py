# file adapted from Jakob Bauer
import math
import tensorflow as tf

import Constants
from datasets.Util.Resize import resize_image, random_crop_image
from datasets.Util.Util import smart_shape

#we do all augmentation in the [0,1] domain!


#TODO: make this compatible with new tensor inputs
class TranslationAugmentor(object):
  """
  Augments the image by translating the content and applying reflection padding.
  """

  def __init__(self, void_label, offset=40):
    """
    Initializes a new instance of the TranslationAugmentor class.

    :param offset: The offset by which the image is randomly translated.
    """
    self.offset = offset
    self.void_label = void_label

  def apply(self, data, offset=None):
    """
    Augments the images by translating the content and applying reflection padding.
    :param data: An array of two elements (images, targets)
    :param offset: the used offset random value (only here for apply_video)
    :return:
    """
    assert False, "currently broken"
    # with tf.name_scope('translate_augmentor'):
    #   # Sample an offset
    #   if offset is None:
    #     offset = tf.random_uniform(minval=-self.offset, maxval=self.offset + 1, dtype=tf.int32,
    #                                shape=[2])
    #
    #   # Extract the image and the label
    #   res_img = self.embed(data[0], offset, 'REFLECT')
    #   res_lbl = self.embed(data[1], offset, self.void_label)
    #
    #   res_img.set_shape(data[0].get_shape())
    #   res_lbl.set_shape(data[1].get_shape())
    #return [res_img, res_lbl]

  @staticmethod
  def embed(image, offset, pad_mode):
    """
    Embeds the image and performs reflection padding.

    :param image: The tensor to translate.
    :param offset: The offset by which we translate.
    :param pad_mode: The padding mode, or a constant
    :return: The augmented image.
    """
    # Compute offsets and sizes
    #shape = image.get_shape().as_list()
    shape = smart_shape(image)
    start = [tf.maximum(-offset[0], 0), tf.maximum(-offset[1], 0)]
    size = [shape[0] - tf.abs(offset[0]), shape[1] - tf.abs(offset[1])]

    # Pad the image on the opposite side
    padding = [
      [tf.maximum(0, offset[0]), tf.maximum(0, -offset[0])],
      [tf.maximum(0, offset[1]), tf.maximum(0, -offset[1])]
    ]

    # no padding on channel dimension for images
    if len(image.get_shape().as_list()) == 3:
      start.append(0)
      size.append(shape[2])
      padding.append([0, 0])

    # Extract the image region that is defined by the offset
    region = tf.slice(image, start, size)
    # region = image[max(-offset[0], 0):shape[0]-max(0, offset[0]),
    #               max(-offset[1], 0):shape[1]-max(0, offset[1])]

    if isinstance(pad_mode, str):
      region = tf.pad(region, padding, mode=pad_mode)
      return region
    else:
      const = pad_mode
      dtype = region.dtype
      region = tf.cast(region, tf.int32) - const
      region = tf.pad(region, padding, mode='CONSTANT')
      region = region + const
      return tf.cast(region, dtype)


class GammaAugmentor(object):
  """
  Performs random gamma augmentation on the first entry of the data array.
  """

  def __init__(self, gamma_range=(-0.1, 0.1)):
    """
    Initializes a new instance of the GammaAugmentor class.
    :param gamma_range: The range from which to sample gamma.
    """
    self.gamma_range = gamma_range

  def apply(self, tensors, factor=None):
    """
    Augments the images. Expects it to be in the [0, 1] range

    :param tensors: dict
    :return: Augmented data
    """
    with tf.name_scope('gamma_augmentor'):
      img = tensors["unnormalized_img"]

      # Sample a gamma factor
      if factor is None:
        factor = tf.random_uniform(shape=[], minval=self.gamma_range[0], maxval=self.gamma_range[1], dtype=tf.float32)
      gamma = tf.log(0.5 + 1 / math.sqrt(2) * factor) / tf.log(0.5 - 1 / math.sqrt(2) * factor)

      # Perform the gamma correction
      aug_image = img ** gamma

      aug_tensors = tensors.copy()
      aug_tensors["unnormalized_img"] = aug_image
    return aug_tensors


class FlipAugmentor(object):
  """
  Augments the data by flipping the image.
  """

  def __init__(self, p=0.5):
    """
    :param p: The probability that the image will be flipped.
    """
    self.p = p

  def apply(self, tensors, doit=None):
    """
    Augments the data.
    """
    with tf.name_scope("flip_augmentor"):
      aug_tensors = tensors.copy()
      if doit is None:
        doit = tf.random_uniform([]) > self.p

      img = tensors["unnormalized_img"]
      img_flipped = tf.image.flip_left_right(img)
      aug_img = tf.cond(doit, lambda: img_flipped, lambda: img)
      aug_tensors["unnormalized_img"] = aug_img

      if "label" in tensors:
        label = tensors["label"]
        label_flipped = tf.reverse(label, axis=[1])
        aug_label = tf.cond(doit, lambda: label_flipped, lambda: label)
        aug_tensors["label"] = aug_label

      if "old_label" in tensors:
        old_label = tensors["old_label"]
        old_label_flipped = tf.reverse(old_label, axis=[1])
        aug_old_label = tf.cond(doit, lambda: old_label_flipped, lambda: old_label)
        aug_tensors["old_label"] = aug_old_label

      if Constants.DT_NEG in tensors:
        u0 = tensors[Constants.DT_NEG]
        u0_flipped = tf.reverse(u0, axis = [1])
        aug_u0 = tf.cond(doit, lambda: u0_flipped, lambda: u0)
        aug_tensors[Constants.DT_NEG] = aug_u0

      if Constants.DT_POS in tensors:
        u1 = tensors[Constants.DT_POS]
        u1_flipped = tf.reverse(u1, axis = [1])
        aug_u1 = tf.cond(doit, lambda: u1_flipped, lambda: u1)
        aug_tensors[Constants.DT_POS] = aug_u1

      if "index_img" in tensors:
        idx_img = tensors["index_img"]
        idx_flipped = tf.reverse(idx_img, axis=[1])
        aug_idx_img = tf.cond(doit, lambda: idx_flipped, lambda: idx_img)
        aug_tensors["index_img"] = aug_idx_img

      if "flow_past" in tensors:
        flow_past = tensors["flow_past"]
        #attention: we also need to negate the x part of the flow
        flow_past_flipped = tf.reverse(flow_past, axis=[1]) * [-1, 1]
        aug_flow_past = tf.cond(doit, lambda: flow_past_flipped, lambda: flow_past)
        aug_tensors["flow_past"] = aug_flow_past

      if "flow_future" in tensors:
        flow_future = tensors["flow_future"]
        #attention: we also need to negate the x part of the flow
        flow_future_flipped = tf.reverse(flow_future, axis=[1]) * [-1, 1]
        aug_flow_future = tf.cond(doit, lambda: flow_future_flipped, lambda: flow_future)
        aug_tensors["flow_future"] = aug_flow_future

    return aug_tensors


#atm only zoom in to avoid having to pad
class ScaleAugmentor(object):
  def __init__(self, void_label):
    self.void_label = void_label

  def apply(self, tensors, scale=None):
    if scale is None:
      # atm minval 1.0 to only zoom in, later also allow smaller values
      scale = tf.random_uniform([1], minval=1.0, maxval=1.25, dtype=tf.float32, seed=None)

    img = tensors["unnormalized_img"]
    h, w = smart_shape(img)[:2]
    crop_size = (h, w)
    h_scaled = tf.to_int32(tf.ceil(tf.cast(h, scale.dtype) * scale))
    w_scaled = tf.to_int32(tf.ceil(tf.cast(w, scale.dtype) * scale))
    scaled_size = tf.concat([h_scaled, w_scaled], axis=0)
    offset = None
    aug_tensors = tensors.copy()

    def _scale(key, bilinear, offset_, force_key=False):
      if force_key:
        assert key in tensors
      if key in tensors:
        im = tensors[key]
        aug_im = resize_image(im, scaled_size, bilinear)
        aug_im, offset_ = random_crop_image(aug_im, crop_size, offset_)
        aug_tensors[key] = aug_im
      return offset_

    offset = _scale("unnormalized_img", True, offset, True)
    _scale("label", False, offset, False)
    _scale("old_label", False, offset)
    _scale("index_img", False, offset)
    _scale("flow_past", True, offset)
    _scale("flow_future", True, offset)
    #attention: when we zoom in, the shift in number of pixels (i.e. optical flow) gets larger
    if "flow_past" in aug_tensors:
      aug_tensors["flow_past"] *= scale
    if "flow_future" in aug_tensors:
      aug_tensors["flow_future"] *= scale

    return aug_tensors


def parse_augmentors(strs, void_label):
  augmentors = []
  for s in strs:
    if s == "gamma":
      augmentor = GammaAugmentor(gamma_range=(-0.05, 0.05))
    elif s == "translation":
      augmentor = TranslationAugmentor(void_label, offset=40)
    elif s == "flip":
      augmentor = FlipAugmentor()
    elif s == "scale":
      augmentor = ScaleAugmentor(void_label)
    else:
      assert False, "unknown augmentor" + s

    augmentors.append(augmentor)
  return augmentors


def apply_augmentors(tensors, augmentors):
  for augmentor in augmentors:
    tensors = augmentor.apply(tensors)
  return tensors
