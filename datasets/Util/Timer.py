from Log import log
import time


class Timer(object):
  def __init__(self, stream=None):
    if stream is None:
      stream = log.v4
    self.stream = stream
    self.start = None

  def __enter__(self):
    self.start = time.time()

  def __exit__(self, exc_type, exc_val, exc_tb):
    end = time.time()
    start = self.start
    self.start = None
    elapsed = end - start
    print >> self.stream, "elapsed", elapsed
