#!/usr/bin/env bash

python preprocess_cityscapes.py --root=/home/eren/Data/Cityscapes/ --rescale=0.5 --output_file=train.h5 --mode=normal --bs=5
