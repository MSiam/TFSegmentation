"""
Rbna yezedak Correctness w ye2lel men falsatk isa
"""

import os
import torchfile
import pickle
import numpy as np
import pdb
import matplotlib.pyplot as plt
import tensorflow as tf

torch_output = torchfile.load('../out_networks_layers/dict_out.t7')

with open('../out_networks_layers/out_linknet_layers.pkl', 'rb') as ff:
    our_output = pickle.load(ff)

print(type(our_output))
print(len(our_output))

#xpl= tf.placeholder(tf.float32, our_output[39].shape)
#pp= tf.pad(xpl, tf.constant([[0,0],[1,1],[1,1],[0,0]]), "CONSTANT")
#pp=tf.nn.max_pool(pp, ksize=(1,3,3,1), strides=(1,2,2,1), padding='VALID')
#pp=tf.image.resize_images(pp, [32,64])
#pp=tf.nn.conv2d()
#session=tf.Session()
#out_temp= session.run(pp, feed_dict={xpl:our_output[2]})
#out_temp2= torch_output[b'network/conv1_x/pool1']
pdb.set_trace()
#plt.figure(1);plt.imshow(our_output[0][0,:,:,0]);
#plt.figure(2);plt.imshow(torch_output[b'x'][0,0,:,:]);plt.show()
#
#plt.figure(1);plt.imshow(our_output[1][0,:,:,0]);
#plt.figure(2);plt.imshow(torch_output[b'network/conv1_x/conv1'][0,0,:,:]);plt.show()
#
#plt.figure(1);plt.imshow(our_output[3][0,:,:,0]);
#plt.figure(2);plt.imshow(torch_output[b'network/conv2_x/conv2_1/conv_1'][0,0,:,:]);plt.show()
#
#plt.figure(1);plt.imshow(our_output[9][0,:,:,0]);
#plt.figure(2);plt.imshow(torch_output[b'network/conv2_x/conv2_2/conv_2'][0,0,:,:]);plt.show()

#plt.figure(1);plt.imshow(our_output[11][0,:,:,0]);
#plt.figure(2);plt.imshow(torch_output[b'network/conv3_x/conv3_1/shortcut_conv'][0,0,:,:]);plt.show()

plt.figure(1);plt.imshow(our_output[37][0,:,:,0]);
plt.figure(2);plt.imshow(torch_output[b'network/conv5_x/conv5_2/bn_2'][0,0,:,:]);plt.show()

plt.figure(1);plt.imshow(our_output[-1][0,:,:,0]);
plt.figure(2);plt.imshow(torch_output[b'network/output_block/deconv_out_2'][0,0,:,:]);plt.show()


print(len(torch_output.items()))

