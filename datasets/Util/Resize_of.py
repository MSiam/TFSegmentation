import tensorflow as tf
from utils import Constants


# enum


class ResizeMode:
  def __init__(self):
    pass

  RandomCrop, ResizeShorterSideToFixedSize, Unchanged, FixedSize, ResizeAndCrop, \
    RandomResizeAndCrop, RandomResize, RandomResizeLess = range(8)


def parse_resize_mode(mode):
  mode = mode.lower()
  if mode == "random_crop":
    return ResizeMode.RandomCrop
  elif mode == "resize_shorter_side_to_fixed_size":
    return ResizeMode.ResizeShorterSideToFixedSize
  elif mode == "unchanged":
    return ResizeMode.Unchanged
  elif mode == "fixed_size":
    return ResizeMode.FixedSize
  elif mode == "resize_and_crop":
    return ResizeMode.ResizeAndCrop
  elif mode == "random_resize_and_crop":
    return ResizeMode.RandomResizeAndCrop
  elif mode == "random_resize":
    return ResizeMode.RandomResize
  elif mode == "random_resize_less":
    return ResizeMode.RandomResizeLess
  else:
    assert False, "Unknonw resize mode " + mode


def resize_image(img, out_size, bilinear):
  if bilinear:
    img = tf.image.resize_images(img, out_size)
  else:
    img = tf.image.resize_nearest_neighbor(tf.expand_dims(img, 0), out_size)
    img = tf.squeeze(img, 0)
  return img


# adapted from code from tf.random_crop
def random_crop_image(img, size, offset=None):
  shape = tf.shape(img)
  check = tf.Assert(
    tf.reduce_all(shape[:2] >= size),
    ["Need value.shape >= size, got ", shape, size])
  with tf.control_dependencies([check]):
    img = tf.identity(img)
  limit = shape[:2] - size + 1
  dtype = tf.int32
  if offset is None:
    offset = tf.random_uniform(shape=(2,), dtype=dtype, maxval=dtype.max, seed=None) % limit
    offset = tf.stack([offset[0], offset[1], 0])
  size0 = size[0] if isinstance(size[0], int) else None
  size1 = size[1] if isinstance(size[1], int) else None
  size_im = tf.stack([size[0], size[1], img.get_shape().as_list()[2]])
  img_cropped = tf.slice(img, offset, size_im)
  out_shape_img = [size0, size1, img.get_shape()[2]]
  img_cropped.set_shape(out_shape_img)
  return img_cropped, offset


def random_crop_tensors(tensors, size):
  tensors_cropped = tensors.copy()
  tensors_cropped["unnormalized_img"], offset = random_crop_image(tensors["unnormalized_img"], size)
  tensors_cropped["flow"], offset = random_crop_image(tensors["flow"], size)
  tensors_cropped["label"], offset = random_crop_image(tensors["label"], size, offset)
  tensors_cropped["raw_label"] = tensors_cropped["label"]
  if "old_label" in tensors:
    tensors_cropped["old_label"], offset = random_crop_image(tensors["old_label"], size, offset)
  if Constants.DT_NEG in tensors:
    tensors_cropped[Constants.DT_NEG], offset = random_crop_image(tensors[Constants.DT_NEG], size, offset)
  if Constants.DT_POS in tensors:
    tensors_cropped[Constants.DT_POS], offset = random_crop_image(tensors[Constants.DT_POS], size, offset)

  return tensors_cropped


def resize(tensors, resize_mode, input_size):
  tensors = tensors.copy()
  if resize_mode == ResizeMode.RandomCrop:
    tensors = random_crop_tensors(tensors, input_size)
  elif resize_mode == ResizeMode.ResizeShorterSideToFixedSize:
    assert len(input_size) == 1
    tensors = resize_shorter_side_fixed_size(tensors, input_size[0])
  elif resize_mode == ResizeMode.Unchanged:
    tensors = resize_unchanged(input_size, tensors)
  elif resize_mode == ResizeMode.FixedSize:
    tensors = resize_fixed_size(tensors, input_size)
  elif resize_mode == ResizeMode.ResizeAndCrop:
    assert len(input_size) == 3
    tensors = resize_shorter_side_fixed_size(tensors, input_size[0])
    tensors = random_crop_tensors(tensors, input_size[1:])
  elif resize_mode == ResizeMode.RandomResizeAndCrop:
    assert len(input_size) in (1, 2)
    if len(input_size) == 2:
      assert input_size[0] == input_size[1]
      crop_size = input_size
    else:
      crop_size = [input_size, input_size]
    tensors = resize_random_scale_with_min_size(tensors, min_size=crop_size)
    tensors = random_crop_tensors(tensors, crop_size)
  elif resize_mode == ResizeMode.RandomResize:
    tensors = resize_random_scale_with_min_size(tensors, min_size=(0, 0))
  elif resize_mode == ResizeMode.RandomResizeLess:
    tensors = resize_random_scale_with_min_size(tensors, min_size=(0, 0), min_scale=0.85, max_scale=1.15)
  else:
    assert False, ("Unknown resize mode", resize_mode)
  return tensors


def resize_random_scale_with_min_size(tensors, min_size, min_scale=0.7, max_scale=1.3):
  assert min_size is not None
  img = tensors["unnormalized_img"]

  h = tf.shape(img)[0]
  w = tf.shape(img)[1]
  shorter_side = tf.minimum(h, w)
  min_scale_factor = tf.cast(min_size, tf.float32) / tf.cast(shorter_side, tf.float32)
  min_scale = tf.maximum(min_scale, min_scale_factor)
  max_scale = tf.maximum(max_scale, min_scale_factor)
  scale_factor = tf.random_uniform(shape=[], minval=min_scale, maxval=max_scale, dtype=tf.float32)
  scaled_size = tf.cast(tf.round(tf.cast(tf.shape(img)[:2], tf.float32) * scale_factor), tf.int32)
  tensors_out = resize_fixed_size(tensors, scaled_size)
  return tensors_out


def resize_fixed_size(tensors, input_size):
  tensors_out = tensors.copy()
  assert input_size is not None
  img = tensors["unnormalized_img"]
  flow = tensors["flow"]
  label = tensors["label"]
  img = resize_image(img, input_size, True)
  flow = resize_image(flow, input_size, True)
  label = resize_image(label, input_size, False)
  if "old_label" in tensors:
    old_label = tensors["old_label"]
    old_label = resize_image(old_label, input_size, False)
    tensors_out["old_label"] = old_label
  if Constants.DT_NEG in tensors:
    u0 = tensors[Constants.DT_NEG]
    u0 = resize_image(u0, input_size, False)
    tensors_out[Constants.DT_NEG] = u0
  if Constants.DT_POS in tensors:
    u1 = tensors[Constants.DT_POS]
    u1 = resize_image(u1, input_size, False)
    tensors_out[Constants.DT_POS] = u1
#    print "Shape of u1: " + u1.get_shape()
  tensors_out["unnormalized_img"] = img
  tensors_out["label"] = label
  tensors_out["flow"] = flow

  #do not change raw_label
  #TODO: this behaviour is different to previous version, check if it breaks anything
  #tensors_out["raw_label"] = label  # raw_label refers to after resizing but before augmentations
  return tensors_out


def resize_shorter_side_fixed_size(tensors, input_size):
  assert input_size is not None
  img = tensors["unnormalized_img"]
  h = tf.shape(img)[0]
  w = tf.shape(img)[1]
  size = tf.constant(input_size)
  h_new = tf.cond(h < w, lambda: size, lambda: h * size / w)
  w_new = tf.cond(h < w, lambda: w * size / h, lambda: size)
  new_shape = tf.stack([h_new, w_new])
  tensors_out = resize_fixed_size(tensors, new_shape)
  return tensors_out


def resize_unchanged(input_size, tensors):
  tensors_out = tensors.copy()
  if input_size is not None:
    img = tensors["unnormalized_img"]
    flow = tensors["flow"]
    label = tensors["label"]
    img.set_shape((input_size[0], input_size[1], None))
    flow.set_shape((input_size[0], input_size[1], None))
    label.set_shape((input_size[0], input_size[1], None))

    def _set_shape(key, n_channels=None):
      if key in tensors:
        tensor = tensors[key]
        tensor.set_shape((input_size[0], input_size[1], n_channels))
        tensors_out[key] = tensor

    _set_shape("old_label", 1)
    _set_shape(Constants.DT_NEG, 2)
    _set_shape(Constants.DT_POS, 2)
    _set_shape("flow_future", 2)
    _set_shape("flow_past", 2)
    _set_shape("index_img", 2)
  return tensors_out
