import os
import numpy as np

x = np.load('xnames_val.npy')
y = []
for i in range(x.shape[0]):
    dir_name = ""
    under_idx = 0
    for j in range(len(x[i])):
        if x[i][j] == '_':
            under_idx = j
            break
        dir_name += x[i][j]
    dot_idx = 0
    for j in range(under_idx, len(x[i])):
        if x[i][j] == '.':
            dot_idx = j
            break
    y.append(dir_name + "/" + dir_name + x[i][under_idx:dot_idx] + "_gtFine_labelIds.png")
