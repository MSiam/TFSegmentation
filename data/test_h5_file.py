import h5py
import numpy as np

filename = 'cscapes_train.h5'
f = h5py.File(filename, 'r')

inputs = f['input'][0:100]
outputs = f['output'][0:100]

print(inputs.shape)
print(inputs.dtype)
print(outputs.shape)
print(outputs.dtype)
