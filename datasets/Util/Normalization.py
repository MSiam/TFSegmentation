import numpy

# IMAGENET_RGB_MEAN = numpy.array((124.0, 117.0, 104.0), dtype=numpy.float32) / 255.0
# values from https://github.com/itijyou/ademxapp/blob/0239e6cf53c081b3803ccad109a7beb56e3a386f/iclass/ilsvrc.py
IMAGENET_RGB_MEAN = numpy.array([0.485, 0.456, 0.406], dtype="float32")
IMAGENET_RGB_STD = numpy.array([0.229, 0.224, 0.225], dtype="float32")


def normalize(img, img_mean=IMAGENET_RGB_MEAN, img_std=IMAGENET_RGB_STD):
  if hasattr(img, "get_shape"):
    l = img.get_shape()[-1]
    if img_mean is not None and l != img_mean.size:
      img_mean = numpy.concatenate([img_mean, numpy.zeros(l - img_mean.size, dtype="float32")], axis=0)
    if img_std is not None and l != img_std.size:
      img_std = numpy.concatenate([img_std, numpy.ones(l - img_std.size, dtype="float32")], axis=0)

  if img_mean is not None:
    img -= img_mean
  if img_std is not None:
    img /= img_std
  return img


def unnormalize(img, img_mean=IMAGENET_RGB_MEAN, img_std=IMAGENET_RGB_STD):
  if hasattr(img, "get_shape"):
    l = img.get_shape()[-1]
    if img_mean is not None and l != img_mean.size:
      img_mean = numpy.concatenate([img_mean, numpy.zeros(l - img_mean.size, dtype="float32")], axis=0)
    if img_std is not None and l != img_std.size:
      img_std = numpy.concatenate([img_std, numpy.ones(l - img_std.size, dtype="float32")], axis=0)

  if img_std is not None:
    img *= img_std
  if img_mean is not None:
    img += img_mean
  return img
