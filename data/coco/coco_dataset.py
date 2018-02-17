from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
import scipy.misc
import numpy as np
import skimage.io as io
#import pdb
import matplotlib.pyplot as plt

class MYCOCO(object):
    alllabels= {1: u'person', 2: u'bicycle', 3: u'car', 4: u'motorcycle', 5: u'airplane', 6: u'bus', 7: u'train', 8: u'truck', 9: u'boat', 10: u'traffic light', 11: u'fire hydrant', 13: u'stop sign', 14: u'parking meter', 15: u'bench', 16: u'bird', 17: u'cat', 18: u'dog', 19: u'horse', 20: u'sheep', 21: u'cow', 22: u'elephant', 23: u'bear', 24: u'zebra', 25: u'giraffe', 27: u'backpack', 28: u'umbrella', 31: u'handbag', 32: u'tie', 33: u'suitcase', 34: u'frisbee', 35: u'skis', 36: u'snowboard',    37: u'sports ball', 38: u'kite', 39: u'baseball bat', 40: u'baseball glove', 41: u'skateboard', 42: u'surfboard', 43: u'tennis racket', 44: u'bottle', 46: u'wine glass', 47: u'cup', 48: u'fork', 49: u'knife', 50: u'spoon', 51: u'bowl', 52: u'banana', 53: u'apple', 54: u'sandwich', 55: u'orange', 56: u'broccoli', 57: u'carrot', 58: u'hot dog', 59: u'pizza', 60: u'donut', 61: u'cake', 62: u'chair', 63: u'couch', 64: u'potted plant', 65: u'bed', 67: u'dining table', 70: u'toilet',    72: u'tv', 73: u'laptop', 74: u'mouse', 75: u'remote', 76: u'keyboard', 77: u'cell phone', 78: u'microwave', 79: u'oven', 80: u'toaster', 81: u'sink', 82: u'refrigerator', 84: u'book', 85: u'clock', 86: u'vase', 87: u'scissors', 88: u'teddy bear', 89: u'hair drier', 90: u'toothbrush'}

    labelID_2_trainID = {}

    def create_label2train(self):
        counter= 1
        MYCOCO.labelID_2_trainID[0]= 0
        for k in MYCOCO.alllabels.keys():
            MYCOCO.labelID_2_trainID[k]= counter
            counter+= 1

    def __init__(self, data_dir, data_type, resize_f, hf):
        self.data_dir= data_dir
        self.data_type= data_type
        self.split= self.data_type[:-4]
        self.ann_file= '{}annotations/instances_{}.json'.format(self.data_dir,self.data_type)
        self.coco=COCO(self.ann_file)
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat2name= {}
        self.resize_f= resize_f
        for c in self.cats:
            self.cat2name[c['id']]= c['name']
        self.hf= hf
        self.create_label2train()

    def read_imgs_annotations(self):
#        catIds = self.coco.getCatIds(catNms=self.interested_cats)
#        imgIds = self.coco.getImgIds(catIds=catIds)
        imgIds = self.coco.getImgIds()
        shape= (len(imgIds), self.resize_f[0], self.resize_f[1],3)
        image_dataset = self.hf.create_dataset('images_' + self.split, shape, dtype=np.uint8)
        label_dataset = self.hf.create_dataset('labels_' + self.split, shape[:-1], dtype=np.uint8)
        counter = 0
        for iid in imgIds:
            img = self.coco.loadImgs(iid)[0]
            I = io.imread('%simages/%s/%s'%(self.data_dir,self.data_type,img['file_name']))
            mask= np.zeros(I.shape[:2], dtype=np.uint8)
            print('handling image ', img['file_name'])
            I= scipy.misc.imresize(I,self.resize_f)
            if len(I.shape)<3:
                continue
            image_dataset[counter, :, :, :]= I
            annIds = self.coco.getAnnIds(imgIds=img['id'], iscrowd=None)
            anns = self.coco.loadAnns(annIds)
            mask= self.parse_mask(anns, mask)
            mask= scipy.misc.imresize(mask,self.resize_f, 'nearest')
            label_dataset[counter, :, :]= mask
#            plt.imshow(I);plt.show()
#            plt.imshow(mask);plt.show()
            counter += 1

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    def parse_mask(self, anns, img_mask):
            """
            Parse the specified annotations.
            :param anns (array of object): annotations to display
            :return: None
            """
            if len(anns) == 0:
                return img_mask
            for ann in anns:
                m = self.annToMask(ann, img_mask.shape[0], img_mask.shape[1])
                img_mask[m==1]= ann['category_id']
            train_mask= np.zeros_like(img_mask)
            for c in MYCOCO.labelID_2_trainID.keys():
                train_mask[img_mask==c]= MYCOCO.labelID_2_trainID[c]
            return train_mask

