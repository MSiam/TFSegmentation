from metrics import Metrics
from os import listdir
from os.path import isfile, join
from scipy.misc import imread

import pdb

gt_imagesDir='/menna/motion_segmentation/DATA/imagepair/testing_1/mask'

output_imagesDir='/menna/motion_segmentation/RUNS/KittiSeg_2stream_imagepair/pred'

metrices=Metrics(2)

gt_images = [f for f in listdir(gt_imagesDir) if isfile(join(gt_imagesDir, f))]
gt_images=sorted(gt_images)

output_images = [f for f in listdir(output_imagesDir) if isfile(join(output_imagesDir, f))]
output_images=sorted(output_images)


for i in range(len(gt_images)):
    print(gt_images[i],output_images[i])
    gt_path=gt_imagesDir+'/'+gt_images[i]
    gt_img=imread(gt_path)>127
    output_path=output_imagesDir+'/'+output_images[i]
    output_img=imread(output_path)>127#for large sclae exps:100,for small scale exps:127
    metrices.update_metrics(output_img,gt_img,0,0)

metrices.compute_rates()
metrices.compute_final_metrics(1)
print("Recall "+str(metrices.rec))
print("Precision "+str(metrices.prec))
print("F score "+str(metrices.fmes))
print("Iou "+ str(metrices.mean_iou_index))






