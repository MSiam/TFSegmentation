import numpy as np
import h5py
import argparse
import os
from scipy import misc
import matplotlib.pyplot as plt

import pdb

def readCityScapes(hf, path_images, path_labels, args, split='train'):
    names =[]
    for root, dirs, files in os.walk(path_images):
        for f in sorted(files):
            if f.split('.')[-1].lower() == 'png':
                names.append(f)
                temp = root
            else:
                continue
    print ("number of found images in \n %s \n is %s"%(path_images, len(names)))

    hf.create_dataset('names_'+split, data=names)
    img = misc.imread(temp+'/'+names[-1])
    h, w, c = img.shape
    if args.rescale is not None:
        h = int(h*args.rescale)
        w = int(w*args.rescale)
    shape = (len(names), c, h, w)

    pdb.set_trace()
    image_dset = hf.create_dataset('images_'+split, shape, dtype=np.uint8)
    i = 0
    for root, dirs, files in os.walk(path_images):
        for f in sorted(files):
            if f.split('.')[-1].lower() == 'png':
                img = misc.imread(root+'/'+f)
                if args.rescale is not None:
                    img = misc.imresize(img, (h,w))
                    plt.imshow(img); plt.show()
                    img = img.transpose(2,0,1)
                if img.shape != (c,h,w):
                    print ("an image is skipped due to inconsistet shape with %s"%str(img.shape))
                    continue
                image_dset[i, :c, :h, :w] = img
                i = i+1
                if i%100 == 1:
                    print ("%sth image has processed"%(str(i)))

    print ("%s images have processes in total"%str(i))

    shape = (len(names),h,w)
    image_dset = hf.create_dataset('labels_'+split, shape, dtype=np.uint8)
    i = 0
    for root, dirs, files in os.walk(path_labels):
        for f in sorted(files):
            if f.split('.')[-1].lower() == 'png' and 'labelIds' in f:
                img = misc.imread(root+'/'+f)
                if args.rescale is not None:
                    img = misc.imresize(img, (h,w))
                if img.shape != (h,w):
                    print ("an image is skipped due to inconsistet shape with %s"%str(img.shape))
                    continue
                image_dset[i, :h, :w] = img
                if i%100 == 1:
                    print ("%sth label has processed"%(str(i)))
                i = i+1
    print ("%s labels was processesed in total"%str(i))


def main(args):
    hf = h5py.File(args.output_file, 'w')
    train_images_path = args.root + 'leftImg8bit/train/darmstadt'
    train_labels_path = args.root + 'gtCoarse/train/darmstadt'
    valid_images_path = args.root + 'leftImg8bit/val/lindau'
    valid_labels_path = args.root + 'gtCoarse/val/lindau'
    readCityScapes(hf, train_images_path, train_labels_path, args, split='train')
    readCityScapes(hf, valid_images_path, valid_labels_path, args, split='valid')
    hf.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default='/data/cityscapes/',
                        help='path to the dataset root that includes leftImg8bit_trainextra leftImg8bit_valid/train and gtCoarse')
    parser.add_argument("--rescale", default=0.25,
                         help="rescale ratio. eg --rescale 0.5")
    parser.add_argument("--output_file", default='leftImg8bit_extra.h5',
                         help="output file for the h5 dataset.")
    args = parser.parse_args()
    main(args)

