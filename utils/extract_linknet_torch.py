import numpy as np
import pickle
import os
import torchfile
from collections import OrderedDict
import pdb

model_weights= torchfile.load('/usr/data/menna/LinkNet-1.0/dict_net.t7')

model_weights= OrderedDict(model_weights)

# Transpose conv and fc weights
for k, v in model_weights.items():
    if len(v.shape) == 4:
        model_weights[k] = np.transpose(v, (2, 3, 1, 0))
    elif len(v.shape) == 2:
        model_weights[k] = np.transpose(v)
    else:
        model_weights[k] = v
    print('name: ' + k + " - shape: " + str(model_weights[k].shape))

pdb.set_trace()
with open('linknet_weights.pkl', "wb") as f:
    pickle.dump(model_weights, f)
