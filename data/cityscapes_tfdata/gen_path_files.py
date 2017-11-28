import numpy as np
import os
import sys
import pdb
import scipy.misc

#def main(main_path, split, out_path):
#
#    path_file= open(out_path, 'w')
#
#    img_dir= main_path+'images/'+split+'/'
#    short_img_dir= 'images/'+split+'/'
#    label_dir= main_path+'labels/'+split+'/'
#    short_label_dir= 'labels/'+split+'/'
#
#    imgs_folders= sorted(os.listdir(img_dir))
#    labels_folders= sorted(os.listdir(label_dir))
#
#    for i in range(len(imgs_folders)):
#        imgs_files= sorted(os.listdir(img_dir+imgs_folders[i]))
#        labels_files= sorted(os.listdir(label_dir+labels_folders[i]))
#
#        for j in range(len(imgs_files)):
#            path_file.write(short_img_dir+imgs_folders[i]+'/'+imgs_files[j]+' '+short_label_dir+labels_folders[i]+'/'+labels_files[j*4+2]+'\n')
#
#
#    path_file.close()
#main(sys.argv[1], sys.argv[2], sys.argv[3])


labelID_2_trainID = {0: 19,  # 'unlabeled'
                     1: 19,  # 'ego vehicle'
                     2: 19,  # 'rectification border'
                     3: 19,  # 'out of roi'
                     4: 19,  # 'static'
                     5: 19,  # 'dynamic'
                     6: 19,  # 'ground'
                     7: 0,  # 'road'
                     8: 1,  # 'sidewalk'
                     9: 19,  # 'parking'
                     10: 19,  # 'rail track'
                     11: 2,  # 'building'
                     12: 3,  # 'wall'
                     13: 4,  # 'fence'
                     14: 19,  # 'guard rail'
                     15: 19,  # 'bridge'
                     16: 19,  # 'tunnel'
                     17: 5,  # 'pole'
                     18: 19,  # 'polegroup'
                     19: 6,  # 'traffic light'
                     20: 7,  # 'traffic sign'
                     21: 8,  # 'vegetation'
                     22: 9,  # 'terrain'
                     23: 10,  # 'sky'
                     24: 11,  # 'person'
                     25: 12,  # 'rider'
                     26: 13,  # 'car'
                     27: 14,  # 'truck'
                     28: 15,  # 'bus'
                     29: 19,  # 'caravan'
                     30: 19,  # 'trailer'
                     31: 16,  # 'train'
                     32: 17,  # 'motorcycle'
                     33: 18,  # 'bicycle'
                     -1: 19,  # 'license plate'
                     }


def custom_ignore_labels(img):
    img_temp = np.zeros(img.shape, dtype=np.uint8)
    for k, v in labelID_2_trainID.items():
        img_temp[img == k] = v
    return img_temp

def save_labels(main_dir, paths_file):
    f= open(paths_file, 'r')

    for line in f:
        tokens= line.strip().split(' ')
        img= scipy.misc.imread(main_dir+tokens[1])
        iimg= custom_ignore_labels(img)
        tokens[1]= tokens[1].replace('labelIds', 'labelIds_proc')
        scipy.misc.imsave(main_dir+tokens[1], iimg)

save_labels(sys.argv[1], sys.argv[2])

