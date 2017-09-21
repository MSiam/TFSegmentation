from abc import abstractmethod

import tensorflow as tf

import Constants
from datasets.Augmentors import parse_augmentors, apply_augmentors
from datasets.Dataset import Dataset
from datasets.Util.Input import assemble_input_tensors
from datasets.Util.Normalization import unnormalize
from datasets.Util.Reader import create_tensor_dict
from datasets.Util.Resize import resize


class FeedImageDataset(Dataset):
  def __init__(self, config, num_classes, void_label, subset, image_size, n_color_channels=3, use_old_label=False,
               flow_into_past=False, flow_into_future=False, use_clicks=False):
    super(FeedImageDataset, self).__init__(subset)
    self._num_classes = num_classes
    self._void_label = void_label
    self.image_size = image_size
    self.config = config
    self.use_summaries = self.config.bool("use_summaries", False)

    self.img_placeholder = tf.placeholder(tf.float32, shape=(None, None, n_color_channels), name="img_placeholder")
    self.label_placeholder = tf.placeholder(tf.uint8, shape=(None, None, 1), name="label_placeholder")
    self.tag_placeholder = tf.placeholder(tf.string, shape=(), name="tag_placeholder")

    self.flow_into_past_placeholder = None
    self.flow_into_future_placeholder = None
    self.old_label_placeholder = None
    self.u0_placeholder = None
    self.u1_placeholder = None
    self.feed_dict = None
    if flow_into_past:
      self.flow_into_past_placeholder = tf.placeholder(tf.float32, shape=(None, None, 2))
    if flow_into_future:
      self.flow_into_future_placeholder = tf.placeholder(tf.float32, shape=(None, None, 2))
    if use_old_label:
      self.old_label_placeholder = tf.placeholder(tf.uint8, shape=(None, None, 1))
    if use_clicks:
      self.u0_placeholder = tf.placeholder(tf.float32, shape=(None, None, 2))
      self.u1_placeholder = tf.placeholder(tf.float32, shape=(None, None, 2))

  def get_label_placeholder(self):
    return self.label_placeholder

  def create_feed_dict(self, img, label, tag, old_label=None, flow_past = None, flow_future = None,
                       u0=None, u1=None):

    tensors = create_tensor_dict(unnormalized_img=img, label=label, tag=tag, old_label=old_label,
                                 flow_past=flow_past,flow_future=flow_future,
                                 u0=u0, u1=u1)

    self.feed_dict = {self.img_placeholder: tensors["unnormalized_img"],
                      self.label_placeholder: tensors["label"],
                      self.tag_placeholder: tensors["tag"]}

    if "old_label" in tensors:
      self.feed_dict[self.old_label_placeholder] = tensors["old_label"]
    if "flow_past" in tensors:
      self.feed_dict[self.flow_into_past_placeholder] = tensors["flow_past"]
    if "flow_future" in tensors:
      self.feed_dict[self.flow_into_future_placeholder] = tensors["flow_future"]
    if Constants.DT_NEG in tensors:
      self.feed_dict[self.u0_placeholder] = tensors[Constants.DT_NEG]
    if Constants.DT_POS in tensors:
      self.feed_dict[self.u1_placeholder] = tensors[Constants.DT_POS]

    return self.feed_dict

  def num_classes(self):
    return self._num_classes

  @abstractmethod
  def num_examples_per_epoch(self):
    pass

  def void_label(self):
    return self._void_label

  def create_input_tensors_dict(self, batch_size):
    use_index_img = self.subset != "train"
    tensors = create_tensor_dict(unnormalized_img=self.img_placeholder, label=self.label_placeholder,
                                 tag=self.tag_placeholder, raw_label=self.label_placeholder,
                                 old_label=self.old_label_placeholder, flow_past=self.flow_into_past_placeholder,
                                 flow_future=self.flow_into_future_placeholder, use_index_img=use_index_img,
                                 u0=self.u0_placeholder, u1=self.u1_placeholder)

    # TODO: need to set shape here?

    resize_mode, input_size = self._get_resize_params(self.subset, self.image_size)
    tensors = resize(tensors, resize_mode, input_size)
    if len(input_size) == 3:
      input_size = input_size[1:]
    tensors = self._prepare_augmented_batch(tensors, batch_size, image_size=input_size)

    if self.use_summaries:
      inputs = tensors["inputs"]
      summ0 = tf.summary.image("inputs", unnormalize(inputs[:, :, :, :3]))
      summ1 = tf.summary.image("labels", tensors["labels"] * 255)  # will only work well for binary segmentation
      self.summaries.append(summ0)
      self.summaries.append(summ1)

    return tensors

  # here a batch always consists only of different augmentations of the same image
  def _prepare_augmented_batch(self, tensors, batch_size, image_size=None):
    if self.subset == "train":
      assert batch_size == 1, "only implemented for 1 so far"
      augmentor_strs = self.config.unicode_list("augmentors_train", [])
      augmentors = parse_augmentors(augmentor_strs, self.void_label())
      for augmentor in augmentors:
        tensors = augmentor.apply(tensors)
      if image_size is None:
        image_size = self.image_size
      tensors = assemble_input_tensors(tensors, image_size)

      # batchify
      keys = tensors.keys()
      tensors = {k: tf.expand_dims(tensors[k], axis=0) for k in keys}
    else:
      augmentor_strs = self.config.unicode_list("augmentors_val", [])
      assert "scale" not in augmentor_strs, "scale augmentation during test time not implemented yet"
      assert "translation" not in augmentor_strs, "translation augmentation during test time not implemented yet"
      augmentors = parse_augmentors(augmentor_strs, self.void_label())
      tensors = [tensors for _ in xrange(batch_size)]
      tensors = [apply_augmentors(t, augmentors) for t in tensors]
      tensors = [assemble_input_tensors(t) for t in tensors]
      # batchify
      keys = tensors[0].keys()
      tensors = {k: tf.stack([t[k] for t in tensors], axis=0) for k in keys}
    return tensors


class OneshotImageDataset(FeedImageDataset):
  def __init__(self, config, num_classes, void_label, subset, image_size, use_old_label=False, flow_into_past=False,
               flow_into_future=False, n_color_channels=3):
    super(OneshotImageDataset, self).__init__(config, num_classes, void_label, subset, image_size,
                                              n_color_channels=n_color_channels, use_old_label=use_old_label,
                                              flow_into_past=flow_into_past, flow_into_future=flow_into_future)
    self._video_idx = None
    self._video_tags = None

  def set_video_idx(self, video_idx):
    self._video_idx = video_idx

  def get_video_idx(self):
    return self._video_idx

  def n_videos(self):
    return len(self._video_tags)

  def video_tag(self, idx=None):
    if idx is None:
      idx = self.get_video_idx()
    return self._video_tags[idx]

  @abstractmethod
  def _get_video_data(self):
    pass

  def num_examples_per_epoch(self):
    assert self._video_idx is not None
    return len(self._get_video_data())

  def feed_dict_for_video_frame(self, frame_idx, with_annotations, old_mask=None):
    tensors = self._get_video_data()[frame_idx].copy()
    feed_dict = {self.img_placeholder: tensors["unnormalized_img"], self.tag_placeholder: tensors["tag"]}
    if with_annotations:
      feed_dict[self.label_placeholder] = tensors["label"]

    assert "old_mask" not in tensors
    if old_mask is not None:
      feed_dict[self.old_label_placeholder] = old_mask

    if "flow_past" in tensors:
      feed_dict[self.flow_into_past_placeholder] = tensors["flow_past"]
    if "flow_future" in tensors:
      feed_dict[self.flow_into_future_placeholder] = tensors["flow_future"]

    return feed_dict

  def label_for_video_frame(self, frame_idx):
    return self._get_video_data()[frame_idx]["label"]
