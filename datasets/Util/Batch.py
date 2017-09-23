import tensorflow as tf


def create_batch(batch_size, *tensors):
  if batch_size == 1:
    batch = [tf.expand_dims(t, axis=0) for t in tensors]
  else:
    batch = tf.train.batch(tensors, batch_size, num_threads=8, capacity=5 * batch_size)
  for t in batch:
    t.set_shape([batch_size] + [None] * (t.get_shape().ndims - 1))
  return batch


def create_batch_dict(batch_size, tensors_dict):
  if batch_size == 1:
    batch = {k: tf.expand_dims(t, axis=0) for k, t in tensors_dict.items()}
  else:
    keys = tensors_dict.keys()
    values = tensors_dict.values()
    values = tf.train.batch(values, batch_size, num_threads=8, capacity=5 * batch_size)
    batch = dict(zip(keys, values))
  for t in batch.values():
    t.set_shape([batch_size] + [None] * (t.get_shape().ndims - 1))
  return batch
