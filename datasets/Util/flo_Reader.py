import numpy


#adapted from http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy
# WARNING: this will work on little-endian architectures (eg Intel x86) only!
def read_flo_file(fp):
  with open(fp, 'rb') as f:
    magic = numpy.fromfile(f, numpy.float32, count=1)
    if 202021.25 != magic:
      print ('Magic number incorrect. Invalid .flo file', fp)
    else:
      w = numpy.fromfile(f, numpy.int32, count=1)[0]
      h = numpy.fromfile(f, numpy.int32, count=1)[0]
      data = numpy.fromfile(f, numpy.float32, count=2 * w * h)
      data2D = data.reshape((h, w, 2))
      return data2D
