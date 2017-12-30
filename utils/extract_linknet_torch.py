import numpy as np
import pickle
import os
import torchfile
from collections import OrderedDict
import pdb

model_weights= torchfile.load('/home/eren/Work/TFSegmentation/pretrained_weights/dict_net.t7')
pdb.set_trace()
# Transpose conv and fc weights
model_weights2= {}
for k, v in model_weights.items():
    if len(v.shape) == 4:
        model_weights2[k.decode("utf-8")] = np.transpose(v, (2, 3, 1, 0))
    else:
        model_weights2[k.decode("utf-8") ] = v
    print('name: ' + str(k) + " - shape: " + str(model_weights2[k.decode("utf-8")].shape))

with open('linknet_weights.pkl', "wb") as f:
    pickle.dump(model_weights2, f)

