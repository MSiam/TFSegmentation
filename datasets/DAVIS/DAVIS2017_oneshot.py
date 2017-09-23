import numpy
import glob
import random
import scipy.ndimage
from PIL import Image

from datasets.FeedDataset import OneshotImageDataset
from datasets.DAVIS.DAVIS import NUM_CLASSES, VOID_LABEL, DAVIS2017_DEFAULT_PATH, read_image_and_annotation_list_2017, \
  group_into_sequences, DAVIS2017_IMAGE_SIZE, DAVIS2017_LUCID_DEFAULT_PATH
from datasets.Util.Util import unique_list
from datasets.Util.Reader import create_tensor_dict
from Log import log

#TODO: avoid code duplication with DAVIS_oneshot, maybe we can use this class for both datasets later?


def _load_frame(im, an):
  sp = im.split("/")
  if "__" in sp[-2]:
    seq_full = sp[-2]
    seq_base = seq_full.split("__")[0]
    im_val = scipy.ndimage.imread(im.replace(seq_full, seq_base)) / 255.0
    if an is None:
      an_postproc = numpy.zeros(im_val.shape[:2], numpy.uint8)
    else:
      #an_raw = scipy.ndimage.imread(an.replace(seq_full, seq_base))
      #an_postproc = (an_raw == color).astype(numpy.uint8).min(axis=2)
      id_ = numpy.array(int(im.split("__")[1].split("/")[-2]))
      #load like this to prevent scipy from converting the palette indices to rgb
      an_raw = numpy.array(Image.open(an.replace(seq_full, seq_base)))
      an_postproc = (an_raw == id_).astype(numpy.uint8)
  else:
    im_val = scipy.ndimage.imread(im) / 255.0
    if an is None:
      an_postproc = numpy.zeros(im_val.shape[:2], numpy.uint8)
    else:
      an_postproc = (numpy.array(Image.open(an)) > 0).astype("uint8")

  an_val = numpy.expand_dims(an_postproc, 2)
  tag_val = im
  tensors = create_tensor_dict(unnormalized_img=im_val, label=an_val, tag=tag_val)
  return tensors


def _load_video(imgs, ans):
  video = []
  for im, an in zip(imgs, ans):
    tensors = _load_frame(im, an)
    video.append(tensors)
  return video


class Davis2017OneshotDataset(OneshotImageDataset):
  def __init__(self, config, subset):
    super(Davis2017OneshotDataset, self).__init__(config, NUM_CLASSES, VOID_LABEL, subset,
                                                  image_size=DAVIS2017_IMAGE_SIZE)
    self._video = None
    data_dir = config.unicode("davis_data_dir", DAVIS2017_DEFAULT_PATH)

    self.split = config.unicode("split", "val")
    assert self.split in ("val", "dev", "eval")
    if self.split == "val":
      list_file = "ImageSets/2017/val.txt"
    elif self.split == "eval":
      list_file = "ImageSets/2017/test-challenge.txt"
    else:
      list_file = "ImageSets/2017/test-dev.txt"
    if not config.bool("adjustable_output_layer", False):
      list_file = list_file.replace(".txt", "_ids.txt")
    imgs, ans = read_image_and_annotation_list_2017(data_dir + list_file, data_dir)
    self._video_tags = unique_list([im.split("/")[-2] for im in imgs])
    self.imgs_seqs = group_into_sequences(imgs)
    self.ans_seqs = group_into_sequences(ans)

    only_first_frame_annotation_available = self.split in ("dev", "eval")
    if only_first_frame_annotation_available:
      self.ans_seqs = [[ans_seq[0]] + ([None] * (len(ans_seq) - 1)) for ans_seq in self.ans_seqs]

    self.use_lucid_data = config.int("lucid_interval", -1) != -1
    self.lucid_data_dir = config.unicode("davis_lucid_data_dir", DAVIS2017_LUCID_DEFAULT_PATH)
    self.lucid_data_video = None
    self.lucid_data_video_current = None

  _cache = None

  _lucid_data_cache = None

  def _get_video_data(self):
    return self._video

  def _load_lucid_data_for_seq(self, video_idx):
    seq = self.video_tag(video_idx)
    assert "__" in seq, "for merged case not implemented yet"
    folder = self.lucid_data_dir + seq
    imgs = glob.glob(folder + "/*.jpg")
    tags = imgs
    masks = [x.replace(".jpg", ".png") for x in imgs]
    imgs = [scipy.ndimage.imread(im) / 255.0 for im in imgs]
    masks = [numpy.expand_dims(scipy.ndimage.imread(mask) / 255, axis=2) for mask in masks]

    #id_ = numpy.array(int(seq.split("__")[1]))
    #masks = [numpy.array(Image.open(mask)) for mask in masks]
    #masks = [(mask == id_).astype(numpy.uint8) for mask in masks]
    #masks = [numpy.expand_dims(mask, 2) for mask in masks]

    tensors = [create_tensor_dict(unnormalized_img=im, label=mask, tag=tag) for im, mask, tag in zip(imgs, masks, tags)]
    return tensors

  def set_video_idx(self, video_idx):
    if self.use_lucid_data and self.get_video_idx() != video_idx:
      if Davis2017OneshotDataset._lucid_data_cache is not None and \
              Davis2017OneshotDataset._lucid_data_cache[0] == video_idx:
        self.lucid_data_video = Davis2017OneshotDataset._lucid_data_cache[1]
      else:
        lucid_data_video = self._load_lucid_data_for_seq(video_idx)
        Davis2017OneshotDataset._lucid_data_cache = (video_idx, lucid_data_video)
        self.lucid_data_video = lucid_data_video

    if self._video is None or self.get_video_idx() != video_idx:
      if Davis2017OneshotDataset._cache is not None and Davis2017OneshotDataset._cache[0] == video_idx:
        self._video = Davis2017OneshotDataset._cache[1]
      else:
        #load video
        print >> log.v2, "loading sequence", (video_idx + 1), self.video_tag(video_idx), "of davis 2017..."
        self._video = _load_video(self.imgs_seqs[video_idx], self.ans_seqs[video_idx])
        print >> log.v2, "done"
        Davis2017OneshotDataset._cache = (video_idx, self._video)
      self._video_idx = video_idx

  def get_number_of_objects_for_video(self):
    video = self._get_video_data()
    an = video[0]["raw_label"]
    n_objects = an.max()
    return n_objects

  def get_lucid_feed_dict(self):
    assert self.lucid_data_video is not None
    assert len(self.lucid_data_video) > 0

    # sampling without replacement (reset when everything was sampled)
    if self.lucid_data_video_current is None or len(self.lucid_data_video_current) == 0:
      self.lucid_data_video_current = self.lucid_data_video[:]
    idx = random.randint(0, len(self.lucid_data_video_current) - 1)
    tensors = self.lucid_data_video_current[idx]
    del self.lucid_data_video_current[idx]
    feed_dict = {self.img_placeholder: tensors["unnormalized_img"], self.tag_placeholder: tensors["tag"],
                 self.label_placeholder: tensors["label"]}
    return feed_dict
