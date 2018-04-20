import numpy as np
import os
import sys
import pdb
import scipy.misc

def main(main_path, split, out_path):

    path_file= open(out_path, 'w')
    img_dir= main_path+'/images/'
    short_img_dir= 'images/'
    mask_dir= main_path+'/mask/'
    short_mask_dir= 'mask/'

    imgs_files= sorted(os.listdir(img_dir))
    labels_files= sorted(os.listdir(mask_dir))

    for j in range(len(imgs_files)):
         path_file.write(short_img_dir+'/'+imgs_files[j]+' '
                 +short_mask_dir+'/'+labels_files[j]+'\n')

    path_file.close()

main(sys.argv[1], '', sys.argv[2])
