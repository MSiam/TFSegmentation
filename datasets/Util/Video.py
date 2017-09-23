import numpy
from Log import log


def shift_im(im, offset):
  return shift(im, offset, "reflect", None)


def shift_lab(lab, offset, void_label):
  return shift(lab, offset, "constant", void_label)


def generate_video(im, lab, n_frames, max_speed, void_label):
  speed = (numpy.random.rand() * 2 - 1.0) * max_speed
  step_offset = numpy.random.rand(2)
  step_offset /= numpy.linalg.norm(step_offset, 2)
  step_offset *= speed

  video_ims = [im]
  video_labs = [lab]
  total_offset = numpy.array([0.0, 0.0])
  for frame in xrange(n_frames - 1):
    total_offset += step_offset
    im_frame = shift_im(im, total_offset)
    lab_frame = shift_lab(lab, total_offset, void_label)
    video_ims.append(im_frame)
    video_labs.append(lab_frame)
  return video_ims, video_labs


def shift(im, offset, mode, value=None):
  assert mode in ("reflect", "constant")
  if mode == "reflect":
    assert value is None
  else:
    assert value is not None

  offset = numpy.round(offset).astype("int32")

  start = numpy.maximum(-offset, 0)
  size = im.shape[:2] - numpy.abs(offset)

  # Extract the image region that is defined by the offset
  im = im[start[0]:start[0] + size[0], start[1]:start[1] + size[1]]

  # Pad the image on the opposite side
  padding = numpy.array([
    [max(0, offset[0]), max(0, -offset[0])],
    [max(0, offset[1]), max(0, -offset[1])],
    [0, 0]
  ])

  if mode == "reflect":
    im = numpy.pad(im, padding, mode)
  else:
    im = numpy.pad(im, padding, mode, constant_values=value)
  return im


def make_chunks(fns, size):
  res = []
  for seq_fns in fns:
    l = len(seq_fns)
    if l < size:
      print >> log.v1, "warning, sequence", seq_fns[0], "too short for chunk size", size
    for i in xrange(l / size):
      chunk = seq_fns[size * i: size * (i + 1)]
      res.append(chunk)
  return res
