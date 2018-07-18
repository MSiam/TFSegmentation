import _pickle as cPickle
from abc import ABCMeta, abstractmethod
from scipy.misc import imsave
import numpy
import tensorflow as tf

# from Log import log
from utils.Measures import compute_iou_for_binary_segmentation, compute_measures_for_binary_segmentation, average_measures
from datasets.Util.pascal_colormap import save_with_pascal_colormap


def merge_multi_samples(ys, idx_imgs, target, verbose=True):
  if target.ndim == 4:
    target = numpy.squeeze(target, axis=-1)
  accumulator = ys[0]
  weights = numpy.ones(accumulator.shape[:2], accumulator.dtype)
  initial_iou = compute_iou_for_binary_segmentation(ys[0].argmax(axis=-1), target)
  print ("iou@1 samples:", initial_iou)
  curr_iou = initial_iou
  for k, (y_val, idx_img_val) in enumerate(zip(ys[1:], idx_imgs[1:])):
    indices = (idx_img_val[:, :, 0], idx_img_val[:, :, 1])
    numpy.add.at(accumulator, indices, y_val)
    numpy.add.at(weights, indices, 1)

    # these four lines are just for iou and can be disabled
    logits = accumulator / weights[:, :, numpy.newaxis]
    y_argmax_val = numpy.argmax(logits, axis=-1)
    curr_iou = compute_iou_for_binary_segmentation(y_argmax_val, target)
    if verbose:
      print ("iou@" + str(k + 2), "samples:", curr_iou)
  if verbose:
    print  ("iou improvement", curr_iou - initial_iou)
  # assert (weights > 0).all()
  logits = numpy.expand_dims(accumulator / weights[:, :, numpy.newaxis], axis=0)
  ys_argmax = numpy.expand_dims(numpy.argmax(logits, axis=-1), axis=3)
  return logits, ys_argmax


class Forwarder(object):
  __metaclass__ = ABCMeta

  def __init__(self, sess,config,model):
    # self.engine = engine
    self.session = sess
    self.config = config
    # self.extractions = self.config.unicode_list("extract", [])
    self.extractions = []
    self.model = model

  @abstractmethod
  def forward(self, network, data, save_results=True, save_logits=False):
    pass


class BasicForwarder(Forwarder):
  def __init__(self, sess,config,model):
    super(BasicForwarder, self).__init__(sess,config,model)
    # self.ignore_first_and_last_results = self.config.bool("ignore_first_and_last_results", True)
    self.ignore_first_and_last_results = True
    # self.ignore_first_result = self.config.bool("ignore_first_result", False)
    self.ignore_first_result = False
    if self.ignore_first_result:
      self.ignore_first_and_last_results = False

  def forward(self, network, data, save_results=True, save_logits=False):
    n_total = data.num_examples_per_epoch()
    n_processed = 0
    targets = network.raw_labels
    ys = network.y_softmax

    # e.g. used for resizing
    ys = self._adjust_results_to_targets(ys, targets)

    measures = []
    while n_processed < n_total:
      n, new_measures, _, _, _ = self._process_forward_minibatch(data, network, save_logits, save_results,
                                                                 targets, ys, n_processed)
      measures += new_measures
      n_processed += n
      print (n_processed, "/", n_total)
    if self.ignore_first_and_last_results:
      measures = measures[1:-1]
    elif self.ignore_first_result:
      measures = measures[1:]

    measures = average_measures(measures)
    if hasattr(data, "video_tag"):
      video_idx = data.get_video_idx()
      print ("sequence", video_idx + 1, data.video_tag(video_idx), measures)
    else:
      print (measures)

  @abstractmethod
  def _adjust_results_to_targets(self, y_softmax, targets):
    pass

  @abstractmethod
  def _process_forward_minibatch(self, data, network, save_logits, save_results, targets, ys,
                                 start_frame_idx):
    pass


class ImageForwarder(BasicForwarder):
  def __init__(self,sess,config,model ):
    super(ImageForwarder, self).__init__(sess,config,model)
    self.eval_chunk_size =-1
    self.n_test_samples = self.config.n_test_samples
    self.adjustable_output_layer = False
    assert self.n_test_samples >= 1, self.n_test_samples

  def _adjust_results_to_targets(self, y_softmax, targets):
    # scale it up!
    return tf.image.resize_images(y_softmax, tf.shape(targets)[1:3])

  def _process_forward_result(self, y_argmax, logit, target, tag, extraction_vals, main_folder, save_results):
    # hack for avoiding storing logits for frames, which are not evaluated
    if "DO_NOT_STORE_LOGITS" in tag:
      logit = None
      tag = tag.replace("_DO_NOT_STORE_LOGITS", "")

    if "__" in tag.split("/")[-2]:
      sp = tag.split("/")
      sp2 = sp[-2].split("__")
      assert len(sp2) == 2
      folder = main_folder + sp2[0] + "/" + sp2[1] + "/"
    else:
      folder = main_folder + tag.split("/")[-2] + "/"
    out_fn = folder + tag.split("/")[-1].replace(".jpg", ".png").replace(".bin", ".png")
    tf.gfile.MakeDirs(folder)

    # TODO: generalize for multiple classes
    measures = compute_measures_for_binary_segmentation(y_argmax, target)
    if save_results:
      if self.adjustable_output_layer:
        save_with_pascal_colormap(out_fn, y_argmax)
      else:
        y_scaled = (y_argmax * 255).astype("uint8")
        imsave(out_fn, numpy.squeeze(y_scaled, axis=2))
      print (out_fn)
    if logit is not None:
      out_fn_logits = out_fn.replace(".png", ".pickle")
      cPickle.dump(logit, open(out_fn_logits, "w"), cPickle.HIGHEST_PROTOCOL)
    for e in extraction_vals:
      assert e.shape[0] == 1  # batch size should be 1 here for now
    for name, val in zip(self.extractions, extraction_vals):
      val = val[0]  # remove batch dimension
      sp = out_fn.replace(".png", ".bin").split("/")
      sp[-1] = name + "_" + sp[-1]
      out_fn_extract = "/".join(sp)
      print (out_fn_extract)
      val.tofile(out_fn_extract)
    return measures

  def _process_forward_minibatch(self, data, network, save_logits, save_results, targets, ys,
                                 start_frame_idx):
    main_folder = "forwarded/" + self.model + "/" + data.subset + "/"
    tf.gfile.MakeDirs(main_folder)

    ys_argmax = tf.arg_max(ys, 3)

    # if len(self.extractions) > 0:
    #   assert len(network.tower_layers) == 1, len(network.tower_layers)
    # extractions = []
    # for e in self.extractions:
    #   layer = network.tower_layers[0][e]
    #   extractions += layer.outputs

    if hasattr(data, "feed_dict_for_video_frames"):
      assert self.eval_chunk_size != -1
      feed_dict = data.feed_dict_for_video_frames(start_frame_idx=start_frame_idx,
                                                  end_frame_idx=start_frame_idx + self.eval_chunk_size,
                                                  with_annotations=True)
      is_feed_dataset = True
    elif hasattr(data, "feed_dict_for_video_frame"):
      feed_dict = data.feed_dict_for_video_frame(frame_idx=start_frame_idx, with_annotations=True)
      is_feed_dataset = True
    elif hasattr(data, "get_feed_dict"):
      feed_dict = data.get_feed_dict()
      is_feed_dataset = True
    else:
      feed_dict = None
      is_feed_dataset = False

    if self.n_test_samples > 1:
      assert is_feed_dataset
      assert len(extractions) == 0, len(extractions)  # currently only supported for single sample
      ys_argmax_val, logits_val, targets_val, tags_val, n = self._run_minibatch_multi_sample(
        feed_dict, ys, targets, network.tags, network.index_imgs)
      extraction_vals = []
    else:
      ys_argmax_val, logits_val, targets_val, tags_val, n, extraction_vals = self._run_minibatch_single_sample(
        feed_dict, ys, ys_argmax, extractions, targets, network.tags, network.n_imgs, save_logits)

    measures = []
    for y_argmax, logit, target, tag in zip(ys_argmax_val, logits_val, targets_val, tags_val):
      measure = self._process_forward_result(y_argmax, logit, target, tag, extraction_vals, main_folder, save_results)
      measures.append(measure)
    return n, measures, ys_argmax_val, logits_val, targets_val

  def _run_minibatch_single_sample(self, feed_dict, ys, ys_argmax, extractions, targets, tags, n_imgs, save_logits):
    ops = [ys_argmax, targets, tags, ys]
    ops += extractions
    results = self.session.run(ops, feed_dict)
    n=1
    ys_argmax_val, targets_val, tags, logits = results[:5]
    extraction_vals = results[5:]
    ys_argmax_val = numpy.expand_dims(ys_argmax_val, axis=3)
    return ys_argmax_val, logits, targets_val, tags, n, extraction_vals

  @staticmethod
  def _flip_if_necessary(y, index_img):
    assert y.shape[0] == 1
    assert index_img.shape[0] == 1
    if all(index_img[0, 0, 0] == [0, 0]):
      flip = False
    elif all(index_img[0, 0, -1] == [0, 0]):
      flip = True
    else:
      assert False, "unexpected index img, probably unsupported augmentors were used during test time"
    if flip:
      return y[:, :, ::-1, :]
    else:
      return y

  def _run_minibatch_multi_sample(self, feed_dict, ys, targets, tags, idx_imgs):
    accumulator, index_img, targets_val, tags_val = self.session.run([ys, idx_imgs, targets, tags], feed_dict)
    accumulator = self._flip_if_necessary(accumulator, index_img)

    for k in range(self.n_test_samples - 1):
      ys_val, index_img = self.session.run([ys, idx_imgs], feed_dict)
      ys_val = self._flip_if_necessary(ys_val, index_img)
      accumulator += ys_val

    logits = accumulator / self.n_test_samples
    ys_argmax_val = numpy.expand_dims(numpy.argmax(logits, axis=-1), axis=3)

    n = 1
    return ys_argmax_val, logits, targets_val, tags_val, n
