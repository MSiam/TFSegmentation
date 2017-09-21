import scipy.ndimage
import time
import numpy
import glob
import random

from Log import log
from datasets.DAVIS.DAVIS import NUM_CLASSES, VOID_LABEL, DAVIS_DEFAULT_PATH, DAVIS_FLOW_DEFAULT_PATH,\
  read_image_and_annotation_list, group_into_sequences, DAVIS_IMAGE_SIZE, DAVIS_LUCID_DEFAULT_PATH
from datasets.FeedDataset import OneshotImageDataset
from datasets.Util.Util import unique_list, load_flow_from_flo
from datasets.Util.Reader import create_tensor_dict


def _load_flow(flow_dir, img_fn, future, flow_as_angle):
  if future:
    flow_fn = flow_dir + "Flow_forward/" + img_fn[img_fn.index("480p"):].replace(".jpg", ".flo")
  else:
    flow_fn = flow_dir + "Flow_backward/" + img_fn[img_fn.index("480p"):].replace(".jpg", ".flo")
  flow = load_flow_from_flo(flow_fn, flow_as_angle)
  return flow


def _load_flows(idx, imgs, shape, flow_dir, flow_into_past, flow_into_future, flow_as_angle):
  flow_past = flow_future = None
  if flow_into_past:
    assert flow_dir is not None
    # for the first frame this won't exist!
    if idx == 0:
      flow_past = numpy.zeros(shape[:-1] + (2,), dtype="float32")

      # load forward flow instead and negate it
      #flow_future = _load_flow(flow_dir, imgs[idx], future=True, flow_as_angle=flow_as_angle)
      #flow_past = -flow_future
    else:
      flow_past = _load_flow(flow_dir, imgs[idx], future=False, flow_as_angle=flow_as_angle)
  if flow_into_future:
    assert flow_dir is not None
    # for the last frame this won't exist!
    if idx == len(imgs) - 1:
      flow_future = numpy.zeros(shape[:-1] + (2,), dtype="float32")

      # load backward flow instead and negate it
      #flow_past = _load_flow(flow_dir, imgs[idx], future=False, flow_as_angle=flow_as_angle)
      #flow_future = -flow_past
    else:
      flow_future = _load_flow(flow_dir, imgs[idx], future=True, flow_as_angle=flow_as_angle)
  return flow_past, flow_future


def _load_frame(idx, im, an, imgs, flow_dir, flow_into_past, flow_into_future, flow_as_angle):
  im_val = scipy.ndimage.imread(im) / 255.0

  flow_past, flow_future = _load_flows(idx, imgs, im_val.shape, flow_dir, flow_into_past, flow_into_future,
                                       flow_as_angle)

  an_raw = scipy.ndimage.imread(an)
  if "adaptation" in an.split("/")[-1]:
    an_postproc = an_raw
    an_postproc[an_raw == 128] = 1
  else:
    an_postproc = an_raw / 255
  an_val = numpy.expand_dims(an_postproc, 2)
  tag_val = im

  tensors = create_tensor_dict(unnormalized_img=im_val, label=an_val, tag=tag_val, flow_past=flow_past,
                               flow_future=flow_future)
  return tensors


def _load_video(imgs, ans, flow_dir=None, flow_into_past=False, flow_into_future=False, flow_as_angle=False):
  video = []
  for idx_, (im_, an_) in enumerate(zip(imgs, ans)):
    tensors_ = _load_frame(idx_, im_, an_, imgs, flow_dir, flow_into_past, flow_into_future, flow_as_angle)
    video.append(tensors_)

  #from joblib import Parallel, delayed
  #video = Parallel(n_jobs=20, backend="threading")(
  #  delayed(_load_frame)(idx_, im_, an_, imgs, flow_dir, flow_into_past, flow_into_future, flow_as_angle)
  #  for idx_, (im_, an_) in enumerate(zip(imgs, ans)))

  return video


class DavisOneshotDataset(OneshotImageDataset):
  def __init__(self, config, subset, use_old_label):
    self.flow_into_past = config.bool("flow_into_past", False)
    self.flow_into_future = config.bool("flow_into_future", False)
    super(DavisOneshotDataset, self).__init__(config, NUM_CLASSES, VOID_LABEL, subset, image_size=DAVIS_IMAGE_SIZE,
                                              use_old_label=use_old_label, flow_into_past=self.flow_into_past,
                                              flow_into_future=self.flow_into_future)
    self.data_dir = config.unicode("davis_data_dir", DAVIS_DEFAULT_PATH)
    self.flow_dir = config.unicode("davis_flow_data_dir", DAVIS_FLOW_DEFAULT_PATH)
    self.flow_as_angle = config.bool("flow_as_angle", False)
    video_range = self.config.int_list("video_range", [])
    if len(video_range) == 0:
      video_range = None
    self.trainsplit = config.int("trainsplit", 0)
    self.adaptation_model = config.unicode("adaptation_model", "")
    self.use_lucid_data = config.int("lucid_interval", -1) != -1
    self.lucid_data_dir = config.unicode("davis_lucid_data_dir", DAVIS_LUCID_DEFAULT_PATH)
    self.lucid_data_video = None
    self.lucid_data_video_current = None

    # approach: load all the images into memory
    # note: always load valid here, as we train on the first valid frame!
    list_file = "ImageSets/480p/trainsplit" + str(self.trainsplit) + "_val.txt" if self.trainsplit > 0 else \
        "ImageSets/480p/val.txt"
    self._video_tags, self._videos = self.load_videos(self.data_dir + list_file,
                                                      self.data_dir, video_range=video_range)
    self._video_idx = None
    self.config = config

  #make sure, only one copy of the data is loaded
  _video_data = None

  _lucid_data_cache = None

  def _load_lucid_data_for_seq(self, video_idx):
    seq = self.video_tag(video_idx)
    folder = self.lucid_data_dir + seq
    imgs = glob.glob(folder + "/*.jpg")
    tags = imgs
    masks = [x.replace(".jpg", ".png") for x in imgs]
    imgs = [scipy.ndimage.imread(im) / 255.0 for im in imgs]
    masks = [numpy.expand_dims(scipy.ndimage.imread(mask) / 255, axis=2) for mask in masks]
    tensors = [create_tensor_dict(unnormalized_img=im, label=mask, tag=tag) for im, mask, tag in zip(imgs, masks, tags)]
    return tensors

  def set_video_idx(self, video_idx):
    if self.use_lucid_data and self._video_idx != video_idx:
      if DavisOneshotDataset._lucid_data_cache is not None and DavisOneshotDataset._lucid_data_cache[0] == video_idx:
        self.lucid_data_video = DavisOneshotDataset._lucid_data_cache[1]
      else:
        lucid_data_video = self._load_lucid_data_for_seq(video_idx)
        DavisOneshotDataset._lucid_data_cache = (video_idx, lucid_data_video)
        self.lucid_data_video = lucid_data_video
    super(DavisOneshotDataset, self).set_video_idx(video_idx)

  def load_videos(self, fn, data_dir, video_range=None):
    load_adaptation_data = self.adaptation_model != ""
    if load_adaptation_data:
      assert self.config.unicode("task") == "offline", self.config.unicode("task")
    elif DavisOneshotDataset._video_data is not None:
      #use this cache only if not doing offline adaptation!
      return DavisOneshotDataset._video_data

    print >> log.v4, "loading davis dataset..."
    imgs, ans = read_image_and_annotation_list(fn, data_dir)
    video_tags = unique_list([im.split("/")[-2] for im in imgs])
    imgs_seqs = group_into_sequences(imgs)
    ans_seqs = group_into_sequences(ans)

    if load_adaptation_data and self.subset == "train":
      #change annotations from groundtruth to adaptation data
      for video_tag, ans in zip(video_tags, ans_seqs):
        for idx in xrange(1, len(ans)):
          ans[idx] = "forwarded/" + self.adaptation_model + "/valid/" + video_tag + ("/adaptation_%05d.png" % idx)

    start = time.time()
    videos = [None] * len(imgs_seqs)
    if video_range is None:
      video_range = [0, len(imgs_seqs)]

    from joblib import Parallel, delayed
    videos[video_range[0]:video_range[1]] = Parallel(n_jobs=20, backend="threading")(delayed(_load_video)(
      imgs, ans, self.flow_dir, self.flow_into_past, self.flow_into_future, self.flow_as_angle) for
                                    (imgs, ans) in zip(imgs_seqs, ans_seqs)[video_range[0]:video_range[1]])

    #videos[video_range[0]:video_range[1]] = [_load_video(
    # imgs, ans, self.flow_dir, self.flow_into_past, self.flow_into_future, self.flow_as_angle) for
    #                               (imgs, ans) in zip(imgs_seqs, ans_seqs)[video_range[0]:video_range[1]]]

    DavisOneshotDataset._video_data = (video_tags, videos)
    end = time.time()
    elapsed = end - start
    print >> log.v4, "loaded davis in", elapsed, "seconds"
    return DavisOneshotDataset._video_data

  def _get_video_data(self):
    return self._videos[self._video_idx]

  def get_lucid_feed_dict(self):
    assert self.lucid_data_video is not None
    assert len(self.lucid_data_video) > 0

    #sampling without replacement (reset when everything was sampled)
    if self.lucid_data_video_current is None or len(self.lucid_data_video_current) == 0:
      self.lucid_data_video_current = self.lucid_data_video[:]
    idx = random.randint(0, len(self.lucid_data_video_current) - 1)
    tensors = self.lucid_data_video_current[idx]
    del self.lucid_data_video_current[idx]
    feed_dict = {self.img_placeholder: tensors["unnormalized_img"], self.tag_placeholder: tensors["tag"],
                 self.label_placeholder: tensors["label"]}
    return feed_dict
