import glob
import tensorflow as tf
from datasets.Dataset import ImageDataset
from datasets.Util.Util import username, unique_list
from datasets.Util.Reader import load_label_default

NUM_CLASSES = 2
VOID_LABEL = 255  # for translation augmentation
DAVIS_DEFAULT_PATH = "/fastwork/" + username() + "/mywork/data/DAVIS/"
DAVIS2017_DEFAULT_PATH = "/fastwork/" + username() + "/mywork/data/DAVIS2017/"
DAVIS_FLOW_DEFAULT_PATH = "/fastwork/" + username() + "/mywork/data/DAVIS_data/"
DAVIS_LUCID_DEFAULT_PATH = "/fastwork/" + username() + "/mywork/data/DAVIS_data/lucid/"
DAVIS2017_LUCID_DEFAULT_PATH = "/fastwork/" + username() + "/mywork/data/DAVIS2017_data/lucid/"
DAVIS_IMAGE_SIZE = (480, 854)
DAVIS2017_IMAGE_SIZE = (480, None)


def read_image_and_annotation_list(fn, data_dir):
  imgs = []
  ans = []
  with open(fn) as f:
    for l in f:
      sp = l.split()
      an = data_dir + sp[1]
      im = data_dir + sp[0]
      imgs.append(im)
      ans.append(an)
  return imgs, ans


def group_into_sequences(fns):
  seqs = unique_list([fn.split("/")[-2] for fn in fns])
  seq_fns = [[fn for fn in fns if fn.split("/")[-2] == seq] for seq in seqs]
  return seq_fns


def get_input_list_file(subset, trainsplit):
  if subset == "train":
    if trainsplit == 0:
      return "ImageSets/480p/train.txt"
    elif trainsplit == 1:
      return "ImageSets/480p/trainsplit_train.txt"
    elif trainsplit == 2:
      return "ImageSets/480p/trainsplit2_train.txt"
    elif trainsplit == 3:
      return "ImageSets/480p/trainsplit3_train.txt"
    else:
      assert False, "invalid trainsplit"
  else:
    if trainsplit == 0:
      return "ImageSets/480p/val.txt"
    elif trainsplit == 1:
      return "ImageSets/480p/trainsplit_val.txt"
    elif trainsplit == 2:
      return "ImageSets/480p/trainsplit2_val.txt"
    elif trainsplit == 3:
      return "ImageSets/480p/trainsplit3_val.txt"
    else:
      assert False, "invalid trainsplit"


class DAVISDataset(ImageDataset):
  def __init__(self, config, subset, coord, fraction=1.0):
    super(DAVISDataset, self).__init__("davis", DAVIS_DEFAULT_PATH, NUM_CLASSES, config, subset, coord,
                                       DAVIS_IMAGE_SIZE, VOID_LABEL, fraction, lambda x: x / 255, ignore_classes=[VOID_LABEL])
    self.trainsplit = config.int("trainsplit", 0)

  def read_inputfile_lists(self):
    assert self.subset in ("train", "valid"), self.subset
    list_file = get_input_list_file(self.subset, self.trainsplit)
    imgs, ans = read_image_and_annotation_list(self.data_dir + list_file, self.data_dir)
    return imgs, ans


###### DAVIS 2017 #####

def postproc_2017_labels(labels):
  return tf.cast(tf.reduce_max(labels, axis=2, keep_dims=True) > 0, tf.uint8)


def get_input_list_file_2017(subset):
  if subset == "train":
    return "ImageSets/2017/train.txt"
  elif subset == "valid":
    return "ImageSets/2017/val.txt"
  else:
    assert False, ("invalid subset", subset)


def read_image_and_annotation_list_2017(fn, data_dir):
  imgs = []
  ans = []
  with open(fn) as f:
    for seq in f:
      seq = seq.strip()
      base_seq = seq.split("__")[0]
      imgs_seq = sorted(glob.glob(data_dir + "JPEGImages/480p/" + base_seq + "/*.jpg"))
      ans_seq = [im.replace("JPEGImages", "Annotations").replace(".jpg", ".png") for im in imgs_seq]
      if "__" in seq:
        ans_seq = [x.replace(base_seq, seq) for x in ans_seq]
        imgs_seq = [x.replace(base_seq, seq) for x in imgs_seq]
      imgs += imgs_seq
      ans += ans_seq
  return imgs, ans


class DAVIS2017Dataset(ImageDataset):
  def __init__(self, config, subset, coord, fraction=1.0):
    super(DAVIS2017Dataset, self).__init__("davis2017", DAVIS2017_DEFAULT_PATH, NUM_CLASSES, config, subset, coord,
                                           DAVIS2017_IMAGE_SIZE, VOID_LABEL, fraction, postproc_2017_labels,
                                           lambda x, y: load_label_default(x, y, channels=3))

  def read_inputfile_lists(self):
    assert self.subset in ("train", "valid"), self.subset
    list_file = get_input_list_file_2017(self.subset)
    imgs, ans = read_image_and_annotation_list_2017(self.data_dir + list_file, self.data_dir)
    return imgs, ans
