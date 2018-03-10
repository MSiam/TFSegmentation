import numpy as np
import h5py
import argparse
import os
from tqdm import tqdm
import scipy.misc as misc
import pdb
from random import shuffle
import h5py

#SIZE= (384, 1248)
#SIZE= (384, 1248)
SIZE= (512, 1024)

def write_image_flow_annotation_pairs(filename_pairs, path, split):
    counter = 0
    imgs= []
    labels= []
    for img_path, annotation_path in tqdm(filename_pairs):
        if not os.path.exists(flow_path):
            print('file ', flow_path, ' doesnt exist ')
            continue

        flo = misc.imread(flow_path)
        flo = misc.imresize(flo, SIZE)
        flows.append(flo)
        img = misc.imread(img_path)
        img = misc.imresize(img, SIZE)
        imgs.append(img)
        annotation = misc.imread(annotation_path)
#        annotation[annotation==1]=0
#        annotation[annotation==3]=0
#        annotation[annotation==2]=1
        annotation[annotation<128]=0
        annotation[annotation>=128]=1
#        annotation[annotation<=150]=0
#        annotation[annotation==255]=0
#        annotation[annotation>150]=1
#        import matplotlib.pyplot as plt
#        plt.imshow(annotation); plt.show()
        annotation = misc.imresize(annotation, SIZE, 'nearest')
        labels.append(annotation)

    np.save(path+'/X_'+split+'.npy', imgs)
    np.save(path+'/Y_'+split+'.npy', labels)

def write_image_annotation_pairs(filename_pairs, path, split):
    counter = 0
    imgs= []
    labels= []
    for img_path, annotation_path in tqdm(filename_pairs):
        img = misc.imread(img_path)
        img = misc.imresize(img, SIZE)
        imgs.append(img)
        annotation = misc.imread(annotation_path)
        annotation[annotation<=128]=0
        annotation[annotation>128]=1
        annotation = misc.imresize(annotation, SIZE, 'nearest')
        labels.append(annotation)

    np.save(path+'/X_'+split+'.npy', imgs)
    np.save(path+'/Y_'+split+'.npy', labels)
    if split=='train':
        mean= np.mean(np.asarray(imgs), axis=0)
        np.save(path+'/mean.npy', mean)

        weights= get_weights(2, labels)
        np.save(path+'/weights.npy', weights)

def get_weights(nclasses, yy):
    label_to_frequency= {}
    for c in range(nclasses):
        class_mask= np.equal(yy, c)
        class_mask= class_mask.astype(np.float32)
        label_to_frequency[c]= np.sum(class_mask)

    #perform the weighing function label-wise and append the label's class weights to class_weights
    class_weights = []
    total_frequency = sum(label_to_frequency.values())
    for label, frequency in label_to_frequency.items():
        class_weight = 1 / np.log(1.02 + (frequency / total_frequency))
        class_weights.append(class_weight)

    return class_weights

def main(args):
    parse_paths(args)

def parse_paths(args_):
    filename_pairs = []

    path_file = open(args.pathfile, 'r')
    for line in path_file:
        tkns= line.strip().split(' ')
        filename_pairs.append((args_.root+tkns[0], args_.root+tkns[1]))

    #shuffle(filename_pairs)

    write_image_annotation_pairs(filename_pairs, args_.out, args_.pathfile.split('_')[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default='/Data/VIVID_DARPA/',
                        help='path to the dataset root')
    parser.add_argument("--pathfile", default='train.txt',
                        help='path to the dataset root')
    parser.add_argument("--out", default='vivid_darpa/',
                        help='name of output tfrecords file')
    args = parser.parse_args()
    main(args)
