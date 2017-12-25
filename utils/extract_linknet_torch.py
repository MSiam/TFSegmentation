import numpy as np
import torchfile

# T7_PATH = '../pretrained_weights/linknet/model-cs-iIoU.net'
#
# print('Open LinkNet torch checkpoint: %s' % T7_PATH)
# o = torchfile.load(T7_PATH)


import tensorflow as tf
import numpy as np
import pickle
import os
import torchfile

o = torchfile.load('boom.t7')

for list in o:
    for x in list:
        print(x.shape)