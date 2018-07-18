import numpy as np
import tensorflow as tf

from utils import Constants
from datasets.Augmentors import apply_augmentors
from datasets.Util.Input import assemble_input_tensors
from datasets.Util.Normalization import normalize
from datasets.Util.Resize import resize
from datasets.Util.Util import load_flow_from_flo, create_index_image, smart_shape

def create_tensor_dict_of(unnormalized_img, flow, label, tag, raw_label=None, old_label=None, flow_past=None, flow_future=None,
                       use_index_img=False, u0=None, u1=None):
  tensors = {"unnormalized_img": unnormalized_img, "flow":flow, "label": label, "tag": tag}
  if raw_label is None:
    tensors["raw_label"] = label
  else:
    tensors["raw_label"] = raw_label
  if old_label is not None:
    tensors["old_label"] = old_label
  if flow_past is not None:
    tensors["flow_past"] = flow_past
  if flow_future is not None:
    tensors["flow_future"] = flow_future
  if u0 is not None:
    tensors[Constants.DT_NEG] = u0
  if u1 is not None:
    tensors[Constants.DT_POS] = u1
  if use_index_img:
    shape = smart_shape(unnormalized_img)
    index_img = create_index_image(shape[0], shape[1])
    tensors["index_img"] = index_img
  return tensors


def create_tensor_dict(unnormalized_img, label, tag, raw_label=None, old_label=None, flow_past=None, flow_future=None,
                       use_index_img=False, u0=None, u1=None):
  tensors = {"unnormalized_img": unnormalized_img, "label": label, "tag": tag}
  if raw_label is None:
    tensors["raw_label"] = label
  else:
    tensors["raw_label"] = raw_label
  if old_label is not None:
    tensors["old_label"] = old_label
  if flow_past is not None:
    tensors["flow_past"] = flow_past
  if flow_future is not None:
    tensors["flow_future"] = flow_future
  if u0 is not None:
    tensors[Constants.DT_NEG] = u0
  if u1 is not None:
    tensors[Constants.DT_POS] = u1
  if use_index_img:
    shape = smart_shape(unnormalized_img)
    index_img = create_index_image(shape[0], shape[1])
    tensors["index_img"] = index_img
  return tensors


def load_label_default(img_path, label_path, channels=1):
  label_contents = tf.read_file(label_path)
  label = tf.image.decode_image(label_contents, channels=channels)
  labels = {"label": label}
  return labels


def load_img_default(img_path):
  img_contents = tf.read_file(img_path)

  img = tf.image.decode_image(img_contents, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img.set_shape((None, None, 3))

  return img


def read_images_from_disk(input_queue, input_size, resize_mode, label_postproc_fn=lambda x: x, augmentors=(),
                          label_load_fn=load_label_default, img_load_fn=load_img_default, of_flag= False):
  im_path = input_queue[0]
  if of_flag:
      flow_path= input_queue[1]
      flow = img_load_fn(img_path=flow_path)
      label_path= input_queue[2]
  else:
      label_path = input_queue[1]

  img = img_load_fn(img_path=im_path)

  labels = label_load_fn(im_path, label_path)
  label = labels['label']

  label = label_postproc_fn(label)
  label.set_shape(img.get_shape().as_list()[:-1] + [1])

  old_label = u0 = u1 = None

  if 'old_label' in labels.keys():
    old_label = labels['old_label']
    old_label.set_shape(img.get_shape().as_list()[:-1] + [1])
  if Constants.DT_NEG in labels.keys() and Constants.DT_POS in labels.keys():
    u0 = labels[Constants.DT_NEG]
    u0.set_shape(img.get_shape().as_list()[:-1] + [1])
    # Create a negative click map, where the click points are denoted as 1 and the rest of it as 0.
    # This would majorly be used to show the clicks in summaries.
    [neg_clicks] = tf.py_func(create_clicks_map, [labels['neg_clicks'], u0], [tf.float32], name="create_click_map")
    neg_clicks.set_shape(img.get_shape().as_list()[:-1] + [1])
    u0 = tf.concat([u0, neg_clicks], axis=2)

    u1 = labels[Constants.DT_POS]
    u1.set_shape(img.get_shape().as_list()[:-1] + [1])
    [pos_clicks] = tf.py_func(create_clicks_map, [labels['pos_clicks'], u1], [tf.float32], name="create_click_map")
    pos_clicks.set_shape(img.get_shape().as_list()[:-1] + [1])
    u1 = tf.concat([u1, pos_clicks], axis=2)

    shape = im_path.get_shape()
    im_path = tf.string_join([im_path, tf.as_string(labels['num_clicks'])], separator=":", name="JoinPath")
    im_path.set_shape(shape)
  if of_flag:
    tensors = create_tensor_dict_of(unnormalized_img=img, flow=flow, label=label,
                               old_label=old_label, u0=u0, u1=u1,
                               tag=im_path, raw_label=label)
  else:
    tensors = create_tensor_dict(unnormalized_img=img, label=label,
                               old_label=old_label, u0=u0, u1=u1,
                               tag=im_path, raw_label=label)

  if of_flag:
      from datasets.Util.Resize_of import resize
      from datasets.Augmentors_of import apply_augmentors
      from datasets.Util.Input_of import assemble_input_tensors
  else:
      from datasets.Util.Resize import resize
      from datasets.Augmentors import apply_augmentors
      from datasets.Util.Input import assemble_input_tensors

  tensors = resize(tensors, resize_mode, input_size)
  tensors = apply_augmentors(tensors, augmentors)
  tensors = assemble_input_tensors(tensors)

  summaries = []

  return tensors, summaries


def create_clicks_map(clicks, dt):
  click_map = np.zeros_like(dt)
  if clicks.shape[0] > 0:
    click_map[clicks[:,0], clicks[:,1]] = 1

  return click_map.astype(np.float32)


def load_image_tensorflow(im_path, jpg=True):
  img_contents = tf.read_file(im_path)
  if jpg:
    img = tf.image.decode_jpeg(img_contents, channels=3)
  else:
    img = tf.image.decode_png(img_contents, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  return img


def load_normalized_image_tensorflow(im_path, jpg=True):
  img = load_image_tensorflow(im_path, jpg)
  img = normalize(img)
  return img


def load_png_mask_tensorflow(path, divide_by_255=True):
  contents = tf.read_file(path)
  mask = tf.image.decode_png(contents, channels=1)
  mask = tf.cast(mask, tf.float32)
  if divide_by_255:
    mask /= 255
  return mask


def load_flow_from_flo_tensorflow(fn, flow_as_angle):
  def my_load_flow(f):
    return load_flow_from_flo(f, flow_as_angle)
  flow, = tf.py_func(my_load_flow, [fn], [tf.float32])
  return flow
