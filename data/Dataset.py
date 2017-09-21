from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf
from skimage.draw import circle

import Constants
from Log import log
from datasets.Augmentors import parse_augmentors
from datasets.Util.Batch import create_batch_dict
from datasets.Util.Normalization import unnormalize
from datasets.Util.Reader import read_images_from_disk, load_label_default, load_img_default
from datasets.Util.Resize import parse_resize_mode, ResizeMode


class Dataset(object):
  __metaclass__ = ABCMeta

  def __init__(self, subset):
    self.summaries = []
    self.subset = subset

  def _get_resize_params(self, subset, input_size_default, resize_mode_default=ResizeMode.Unchanged):
    if subset == "valid":
      subset = "val"
    resize_mode_string = self.config.unicode("resize_mode_" + subset, "")
    if resize_mode_string == "":
      resize_mode_string = self.config.unicode("resize_mode_train", "")
    if resize_mode_string == "":
      resize_mode = resize_mode_default
    else:
      resize_mode = parse_resize_mode(resize_mode_string)
    input_size = self.config.int_list("input_size_" + subset, [])
    if len(input_size) == 0:
      input_size = input_size_default
    if resize_mode == ResizeMode.RandomResize:
      input_size = [None, None]
    return resize_mode, input_size

  @abstractmethod
  def num_classes(self):
    pass

  @abstractmethod
  def num_examples_per_epoch(self):
    pass

  # should contain inputs, labels, tags, maybe raw_labels, index_img
  @abstractmethod
  def create_input_tensors_dict(self, batch_size):
    pass

  @abstractmethod
  def void_label(self):
    pass


class ImageDataset(Dataset):
  def __init__(self, dataset_name, default_path, num_classes, config, subset, coord, image_size, void_label=255,
               fraction=1.0, label_postproc_fn=lambda x: x, label_load_fn=load_label_default,
               img_load_fn=load_img_default, ignore_classes=[]):
    super(ImageDataset, self).__init__(subset)
    self._num_classes = num_classes
    self._void_label = void_label
    assert subset in ("train", "valid"), subset
    self.config = config
    self.data_dir = config.unicode(dataset_name + "_data_dir", default_path)
    self.coord = coord
    self.image_size = image_size
    self.inputfile_lists = None
    self.fraction = fraction
    self.label_postproc_fn = label_postproc_fn
    self.label_load_fn = label_load_fn
    self.img_load_fn = img_load_fn
    self.use_summaries = self.config.bool("use_summaries", False)
    self.ignore_classes = ignore_classes

  def _load_inputfile_lists(self):
    if self.inputfile_lists is not None:
      return
    self.inputfile_lists = self.read_inputfile_lists()
    # make sure all lists have the same length
    assert all([len(l) == len(self.inputfile_lists[0]) for l in self.inputfile_lists])
    if self.fraction < 1.0:
      n = int(self.fraction * len(self.inputfile_lists[0]))
      self.inputfile_lists = [l[:n] for l in self.inputfile_lists]

  def _parse_augmentors_and_shuffle(self):
    if self.subset == "train":
      shuffle = True
      augmentor_strs = self.config.unicode_list("augmentors_train", [])
      augmentors = parse_augmentors(augmentor_strs, self.void_label())
      if len(augmentors) == 0:
        print >> log.v1, "warning, no data augmentors used on train"
    else:
      shuffle = False
      augmentor_strs = self.config.unicode_list("augmentors_val", [])
      augmentors = parse_augmentors(augmentor_strs, self.void_label())
    return augmentors, shuffle

  def create_input_tensors_dict(self, batch_size):
    self._load_inputfile_lists()
    resize_mode, input_size = self._get_resize_params(self.subset, self.image_size, ResizeMode.Unchanged)
    augmentors, shuffle = self._parse_augmentors_and_shuffle()

    inputfile_tensors = [tf.convert_to_tensor(l, dtype=tf.string) for l in self.inputfile_lists]
    queue = tf.train.slice_input_producer(inputfile_tensors, shuffle=shuffle)

    tensors_dict, summaries = self._read_inputfiles(queue, resize_mode, input_size, augmentors)
    tensors_dict = create_batch_dict(batch_size, tensors_dict)

    if self.use_summaries:
      inputs = tensors_dict["inputs"]
      input_img = unnormalize(inputs[:, :, :, :3])
      # Add clicks to the image so that they can be viewed in tensorboard
      if inputs.get_shape()[-1] > 4:
        [input_img] = tf.py_func(self.add_clicks, [tf.concat([input_img, inputs[:, :, :, 3:4]],
                                                             axis=3),'r'], [tf.float32])
        [input_img] = tf.py_func(self.add_clicks, [tf.concat([input_img, inputs[:, :, :, 4:5]],
                                                             axis=3), 'g'], [tf.float32])

      summ0 = tf.summary.image("inputs", input_img)
      summ1 = tf.summary.image("labels", tensors_dict["labels"] * 255)  # will only work well for binary segmentation
      summaries = [summ0, summ1]

      # count is incremented after each summary creation. This helps us to keep track of the number of channels
      count = 0

      # Old label would be present either as a single channel, or along with 2 other distance transform channels.
      if inputs.get_shape()[-1] == 4 or inputs.get_shape == 6:
        summ2 = tf.summary.image("old_labels", inputs[:, :, :, 3:4] * 255)
        summaries.append(summ2)
        count += 1
      # Append the distance transforms, if they are available.
      if inputs.get_shape()[-1] > 4:
        # Get negative distance transform from the extra input channels.
        start = 3 + count
        end = start + 1
        start = tf.constant(start)
        end = tf.constant(end)
        summ3 = tf.summary.image(Constants.DT_NEG, inputs[:, :, :, 3:4])

        start = end
        end = start + 1
        summ4 = tf.summary.image(Constants.DT_POS, inputs[:, :, :, 4:5])
        summaries.append(summ3)
        summaries.append(summ4)

    self.summaries += summaries
    return tensors_dict

  # default implementation, should in many cases be overwritten
  def _read_inputfiles(self, queue, resize_mode, input_size, augmentors):
    tensors, summaries = read_images_from_disk(queue, input_size, resize_mode, label_postproc_fn=self.label_postproc_fn,
                                               augmentors=augmentors, label_load_fn=self.label_load_fn,
                                               img_load_fn=self.img_load_fn)
    return tensors, summaries

  def add_clicks(self, inputs, c):
    out_imgs = None
    # pdb.set_trace()

    for input in inputs:
      img = input[:,:,:3]
      # Radius of the point to be diplayed
      r=3
      pts = np.where(input[:,:,3] == 0)
      pts_zipped = zip(pts[0], pts[1])
      if len(pts[0]) > 0:
       for pt in pts_zipped:
         if r < pt[0] < img.shape[0] - r and r < pt[1] < img.shape[1] - r:
            rr, cc = circle(pt[0], pt[1], 5, img.shape)
            img[rr, cc, :] = [np.max(img), np.min(img), np.min(img)] if c == 'r' \
                              else [np.min(img), np.min(img), np.max(img)]

      img = img[np.newaxis, :, :, :]
      if out_imgs is None:
        out_imgs = img
      else:
        out_imgs = np.concatenate((out_imgs, img), axis = 0)

    return out_imgs.astype(np.float32)

  @abstractmethod
  def read_inputfile_lists(self):
    pass

  def num_examples_per_epoch(self):
    self._load_inputfile_lists()
    return len(self.inputfile_lists[0])

  def num_classes(self):
    return self._num_classes

  def void_label(self):
    return self._void_label

  def ignore_classes(self):
    return self.ignore_classes
