import numpy
import math
import tensorflow as tf
from collections import namedtuple
TowerSetup = namedtuple("TowerSetup", ["dtype", "gpu", "is_main_train_tower", "is_training",
                                       "freeze_batchnorm", "variable_device", "use_update_ops_collection",
                                       "batch_size"])


def conv2d(x, W, strides=None):
  if strides is None:
    strides = [1, 1]
  return tf.nn.conv2d(x, W, strides=[1] + strides + [1], padding="SAME")


def conv2d_dilated(x, W, dilation):
  res = tf.nn.atrous_conv2d(x, W, dilation, padding="SAME")
  shape = x.get_shape().as_list()
  shape[-1] = W.get_shape().as_list()[-1]
  res.set_shape(shape)
  return res


def max_pool(x, shape, strides=None):
  if strides is None:
    strides = shape
  return tf.nn.max_pool(x, ksize=[1] + shape + [1],
                        strides=[1] + strides + [1], padding="SAME")


def avg_pool(x, shape):
  return tf.nn.avg_pool(x, ksize=[1] + shape + [1],
                        strides=[1] + shape + [1], padding="VALID")
  #TODO: maywe be should change this to SAME


def global_avg_pool(x):
  assert len(x.get_shape()) == 4
  return tf.reduce_mean(x, [1, 2])


def apply_dropout(inp, dropout):
  if dropout == 0.0:
    return inp
  else:
    keep_prob = 1.0 - dropout
    return tf.nn.dropout(inp, keep_prob)


def prepare_input(inputs):
  #assert len(inputs) == 1, "Multiple inputs not yet implemented"
  if True:
    inp = inputs
    dim = int(inp.get_shape()[-1])
  else:
    dims = [int(inp.get_shape()[3]) for inp in inputs]
    dim = sum(dims)
    inp = tf.concat_v2(inputs, 3)
  return inp, dim


def prepare_collapsed_input_and_dropout(inputs, dropout):
  assert len(inputs) == 1, "Multiple inputs not yet implemented"
  inp = inputs[0]
  shape = inp.get_shape()
  if len(shape) == 4:
    dim = int(numpy.prod(shape[1:4]))
    inp = tf.reshape(inp, [-1, dim])
  else:
    dim = int(shape[-1])
  if dropout != 0.0:
    keep_prob = 1.0 - dropout
    inp = tf.nn.dropout(inp, keep_prob)
  return inp, dim


activs = {"relu": tf.nn.relu, "linear": lambda x: x, "elu": tf.nn.elu}


def get_activation(act_str):
  assert act_str.lower() in activs, "Unknown activation function " + act_str
  return activs[act_str.lower()]


def create_batch_norm_vars(n_out, scope_name="bn"):
  with tf.variable_scope(scope_name):
    initializer_zero = tf.constant_initializer(0.0)
    beta = tf.get_variable("beta", [n_out], initializer_zero)
    initializer_gamma = tf.constant_initializer(1.0)
    gamma = tf.get_variable("gamma", [n_out], initializer_gamma)
    mean_ema = tf.get_variable("mean_ema", [n_out], initializer_zero, trainable=False)
    var_ema = tf.get_variable("var_ema", [n_out], initializer_zero, trainable=False)
    return beta, gamma, mean_ema, var_ema


#adapted from https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/fcn16_vgg.py
def create_bilinear_upsampling_weights(shape):
  height, width = shape[0], shape[1]
  f = math.ceil(width / 2.0)
  c = (2 * f - 1 - f % 2) / (2.0 * f)
  bilinear = numpy.zeros([shape[0], shape[1]])
  for x in range(width):
    for y in range(height):
      value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
      bilinear[x, y] = value
  weights = numpy.zeros(shape)
  for i in range(shape[2]):
    weights[:, :, i, i] = bilinear
  return weights


#adapted from Jakob Bauer
def iou_from_logits(logits, labels):
  """
  Computes the intersection over union (IoU) score for given logit tensor and target labels
  :param logits: 4D tensor of shape [batch_size, height, width, num_classes]
  :param labels: 3D tensor of shape [batch_size, height, width] and type int32 or int64
  :return: 1D tensor of shape [num_classes] with intersection over union for each class, averaged over batch
  """

  with tf.variable_scope("IoU"):
    # compute predictions
    preds = tf.arg_max(logits, dimension=3)

    num_labels = logits.get_shape().as_list()[-1]
    IoUs = []
    for label in range(num_labels):
      # find pixels with given label
      P = tf.equal(preds, label)
      L = tf.equal(labels, label)

      # Union
      U = tf.logical_or(P, L)
      U = tf.reduce_sum(tf.cast(U, tf.float32))

      # intersection
      I = tf.logical_and(P, L)
      I = tf.reduce_sum(tf.cast(I, tf.float32))

      IoUs.append(I / U)

    return tf.reshape(tf.stack(IoUs), (num_labels,))


def upsample_repeat(x, factor=2):
  #(batch, height, width, feat) -> (batch, height, 1, width, feat) -> (batch, height, 1, width, 1, feat)
  #-> (batch, height, 2, width, 2, feat) -> (batch, 2 * height, 2 * width, feat)
  s = tf.shape(x)
  s2 = x.get_shape().as_list()
  x = tf.expand_dims(x, 2)
  x = tf.expand_dims(x, 4)
  x = tf.tile(x, [1, 1, factor, 1, factor, 1])
  x = tf.reshape(x, [s[0], factor * s[1], factor * s[2], s[3]])
  if s2[1] is not None:
    s2[1] *= factor
  if s2[2] is not None:
    s2[2] *= factor
  x.set_shape(s2)
  return x
