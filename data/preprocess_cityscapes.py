import numpy as np
import h5py
import argparse
import os
from scipy import misc
import matplotlib.pyplot as plt
from tqdm import tqdm
#import pdb
def main(args):
    hf = h5py.File(args.output_file, 'w')
    for d in os.listdir(args.root+'images/'):
        if d=='test':
            continue
        train_images_path = args.root + 'images/'+d
        train_labels_path = args.root + 'labels/'+d
        custom_read_cityscape(hf, train_images_path, train_labels_path, args, split=d)
    hf.close()

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


def custom_ignore_labels(img, h, w):
    for i in range(h):
        for j in range(w):
            img[i, j] = labelID_2_trainID[img[i, j]]
    return img


def custom_read_cityscape(hf, path_images, path_labels, args_, split='train'):
    names = []
    root = path_images

    # Read all image files in the path_images directory
    for d in os.listdir(path_images):
        for _, dirs, files in os.walk(path_images+'/'+d):
            for f in sorted(files):
                if f.split('.')[-1].lower() == 'png':
                    names.append(d+'/'+f)
                else:
                    continue

    np.save('names_'+split, np.array(names))
    # Print statistics
    print("number of found images in \n %s is %s img" % (path_images, len(names)))
    print(root)

    # read an image to get the shape
    img = misc.imread(root + '/' + names[-1])
    h, w, c = img.shape

    # rescale the shape
    if args_.crop != 0:
        h = int(h * args_.crop)
        w = int(w * args_.crop)
        args_.rescale= None
        shape = (4*len(names), h, w, c)
    elif args_.rescale is not None:
        h = int(h * args_.rescale)
        w = int(w * args_.rescale)
        shape = (len(names), h, w, c)
    print("Shape of images is %s" % (str(shape)))

    # Create a dataset for images
    image_dataset = hf.create_dataset('images_' + split, shape, dtype=np.uint8)
    # loop on images and scale and save
    i = 0
    for f in tqdm(names):
        img = misc.imread(root + '/' + f)
        if args_.crop !=0:
            img0= img[h:, :w]
            img1= img[h:, w:]
            img2= img[:h, w:]
            img= img[:h,:w]
        elif args_.rescale is not None:
            img = misc.imresize(img, (h, w))
        if img.shape != (h, w, c):
            print("an image is skipped due to inconsistet shape with %s" % str(img.shape))
            continue
        image_dataset[i, :h, :w, :c] = img
        if args.crop !=0 :
            image_dataset[i+1, :h, :w, :c] = img0
            image_dataset[i+2, :h, :w, :c] = img1
            image_dataset[i+3, :h, :w, :c] = img2
            i+= 3

        i = i + 1

    print("%s images have processes in total" % str(i))

    # Save a numpy
    if args_.crop!=0:
        assert image_dataset.shape == (4*len(names), h, w, c)
    else:
        assert image_dataset.shape == (len(names), h, w, c)
    assert image_dataset.dtype == np.uint8
    bs= image_dataset.shape[0] - image_dataset.shape[0]%int(args_.bs)

    image_dataset= image_dataset[:bs,:,:,:]
    np.save('X_'+split, image_dataset)

    if args_.crop!=0:
        shape= (4*len(names),h,w)
    else:
        shape = (len(names), h, w)
    print("Shape of labels is %s" % (str(shape)))

    # Read all image files in the path_images directory
    names = []
    for d in os.listdir(path_images):
        for _, dirs, files in os.walk(path_labels+'/'+d):
            for f in sorted(files):
                if f.split('.')[-1].lower() == 'png' and 'labelIds' in f:
                    names.append(d+'/'+f)
                else:
                    continue

    np.save('ynames_'+split, np.array(names))
    # Create a dataset for labels
    image_dataset = hf.create_dataset('labels_' + split, shape, dtype=np.uint8)
    root = path_labels
    i = 0
    for f in tqdm(names):
        img = misc.imread(root + '/' + f)
        if args_.crop !=0:
            img0= img[h:, :w]
            img0 = custom_ignore_labels(img0, h, w)
            img1= img[h:, w:]
            img1 = custom_ignore_labels(img1, h, w)
            img2= img[:h, w:]
            img2 = custom_ignore_labels(img2, h, w)
            img= img[:h,:w]
            img = custom_ignore_labels(img, h, w)
        elif args_.rescale is not None:
            img = custom_ignore_labels(img, img.shape[0], img.shape[1])
            img = misc.imresize(img, (h, w), 'nearest')
        if img.shape != (h, w):
            print("an image is skipped due to inconsistent shape with %s" % str(img.shape))
            continue
        image_dataset[i, :h, :w] = img
        if args.crop !=0 :
            image_dataset[i+1, :h, :w] = img0
            image_dataset[i+2, :h, :w] = img1
            image_dataset[i+3, :h, :w] = img2
            i+= 3
        i = i + 1
    print("%s labels was processed in total" % str(i))

    # Save a txt to check that everything is ok
    #np.savetxt('Y.txt', np.array(image_dataset[0]), fmt="%d")
    # Save a numpy
    image_dataset= image_dataset[:bs,:,:]
    np.save('Y_'+split, np.array(image_dataset))

def custom_main(args_):
    hf = h5py.File(args_.output_file, 'w')
    train_images_path = args_.root + 'images/aachen'
    train_labels_path = args_.root + 'labels/aachen'
    custom_read_cityscape(hf, train_images_path, train_labels_path, args_, split='train')
    hf.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default='/data/cityscapes/',
                        help='path to the dataset root that includes leftImg8bit_trainextra leftImg8bit_valid/train and gtCoarse')
    parser.add_argument("--crop", default=0, type=float,
                        help="rescale ratio. eg --crop 0")
    parser.add_argument("--rescale", default=0.25, type=float,
                        help="rescale ratio. eg --rescale 0.5")
    parser.add_argument("--output_file", default='leftImg8bit_extra.h5',
                        help="output file for the h5 dataset.")
    parser.add_argument("--mode", default="normal", help="Mode of preparation")
    parser.add_argument("--bs", default=5, help="Batch Size to output divisible number")

    args = parser.parse_args()
    if args.mode == "normal":
        main(args)
    elif args.mode == "custom":
        custom_main(args)
    else:
        print("Please choose a proper mode first THANKS.")
        exit(-1)
