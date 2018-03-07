import numpy as np
import sys
import os
import scipy.misc as misc
import matplotlib.pyplot as plt
import pdb

trainID_2_labelID = {19: 0,  # 'unlabeled'
                     0: 7,  # 'road'
                     1: 8,  # 'sidewalk'
                     2: 11,  # 'building'
                     3: 12,  # 'wall'
                     4: 13,  # 'fence'
                     5: 17,  # 'pole'
                     6: 19,  # 'traffic light'
                     7: 20,  # 'traffic sign'
                     8: 21,  # 'vegetation'
                     9: 22,  # 'terrain'
                     10: 23,  # 'sky'
                     11: 24,  # 'person'
                     12: 25,  # 'rider'
                     13: 26,  # 'car'
                     14: 27,  # 'truck'
                     15: 28,  # 'bus'
                     16: 31,  # 'train'
                     17: 32,  # 'motorcycle'
                     18: 33,  # 'bicycle'
                     }

def postprocess(pred):
    pred_proc= np.zeros(pred.shape, np.uint8)
    for k,v in trainID_2_labelID.items():
        pred_proc[pred==k]= v
    return pred_proc

def main():
    ynames= np.load('full_cityscapes_res/ynames_val.npy')
    for i in range(500):
        pred= np.load(sys.argv[1]+'npy/'+str(i)+'.npy')
        pred= postprocess(pred)
        misc.imsave(sys.argv[1]+'preds/'+str(i)+'.png', pred)
        gt= misc.imresize(misc.imread('/home/eren/Data/Cityscapes/labels/val/'+ynames[i]),(512,1024))
        #plt.imshow(pred);plt.show()
        #plt.imshow(gt);plt.show()
        #pdb.set_trace()
        misc.imsave(sys.argv[1]+'tempgt/'+str(i)+'.png', gt)

if __name__=="__main__":
    main()
