"""
Rbna yezedak Correctness w ye2lel men falsatk isa
"""

import os
import torchfile
import pickle
import numpy as np
import pdb

torch_output = torchfile.load('../out_networks_layers/dict_out.t7')

with open('../out_networks_layers/out_linknet_layers.pkl', 'rb') as ff:
    our_output = pickle.load(ff, encoding='latin1')

print(type(our_output))
print(len(our_output))

print(len(torch_output.items()))

"""
Our output contains a list with all outputs MTRTBEN
"""
pdb.set_trace()

