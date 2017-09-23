import tensorflow as tf

from utils import Constants
from datasets.Util.Normalization import normalize


def assemble_input_tensors(tensors, img_size=(None, None)):
  img_size = list(img_size)
  assert "unnormalized_img" in tensors
  assert "label" in tensors
  assert "tag" in tensors

  img = tensors["unnormalized_img"]
  img = normalize(img)
  label = tensors["label"]
  if "raw_label" in tensors:
    raw_label = tensors["raw_label"]
  else:
    raw_label = label
  tag = tensors["tag"]
  concats = [img]
  n_input_channels = 3
  if "old_label" in tensors:
    concats.append(tf.cast(tensors["old_label"], tf.float32))
    n_input_channels += 1
  if Constants.DT_NEG in tensors:
    # Do not use the click channel as they can be deciphered from the distance transforms.
    u0 = tensors[Constants.DT_NEG][:, :, 0:1]
    u0 = tf.cast(u0, tf.float32)
    clip_value = tf.ones_like(u0)*255
    u0 = tf.where(tf.greater(u0, 255), clip_value, u0 )
    u0 = u0 / 255.0
    concats.append(u0)
    n_input_channels += 1
  if Constants.DT_POS in tensors:
    u1 = tensors[Constants.DT_POS][:, :, 0:1]
    u1 = tf.cast(u1, tf.float32)
    clip_value = tf.ones_like(u1)*255
    u1 = tf.where(tf.greater(u1, 255), clip_value, u1 )
    u1 = u1 / 255.0
    concats.append(u1)
    n_input_channels += 1
  if "flow_past" in tensors:
    concats.append(tensors["flow_past"])
    n_input_channels += 2
  if "flow_future" in tensors:
    concats.append(tensors["flow_future"])
    n_input_channels += 2
  if len(concats) > 1:
    img = tf.concat(concats, axis=2)

  img.set_shape(img_size + [n_input_channels])

  tensors_out = {"inputs": img, "labels": label, "raw_labels": raw_label, "tags": tag}
  if "index_img" in tensors:
    tensors_out["index_imgs"] = tensors["index_img"]
  return tensors_out
