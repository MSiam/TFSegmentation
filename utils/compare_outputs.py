"""
Rbna yezedak Correctness w ye2lel men falsatk isa
"""

import os
import torchfile
import pickle
import numpy as np

torch_output = torchfile.load('../out_networks_layers/dict_net.t7')

with open('../out_networks_layers/out_linknet_layers.pkl', 'rb') as ff:
    our_output = pickle.load(ff, encoding='latin1')