#!/usr/bin/env bash

#python3 main.py --load_config=test.yaml overfit Train FCN8s

# FCN8s exp runs
#python main.py --load_config=fcn8s_exp_test.yaml test Train FCN8s
#python main.py --load_config=fcn8s_exp_video.yaml test Train FCN8s

# VGG16UNET exp runs
python main.py --load_config=vgg16unet_exp_test.yaml overfit Train VGG16UNET