
import h5py
import numpy as np

filename = 'full_cityscapes/cscapes_train.h5'
f = h5py.File(filename, 'r')

inputs = f['X'][[0,1,20,500,5000]]
outputs = f['Y'][[0,1,20,500,5000]]

print(inputs.shape)
print(inputs.dtype)
print(outputs.shape)
print(outputs.dtype)
