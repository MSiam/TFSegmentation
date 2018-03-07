#!/usr/bin/env bash

#python preprocess_cityscapes.py --root=/home/eren/Data/Cityscapes/ --crop=0.5 --rescale=0.5 --output_file=train.h5 --mode=normal --bs=5
#python preprocess_cityscapes.py --root=/home/eren/Data/Cityscapes/ --rescale=0.5 --output_file=train.h5 --mode=normal --bs=5
python preprocess_cityscapes_h5.py --root=../../Cityscapes/ --rescale=0.5 --dir=train_extra --out=cscapes_train.h5
python preprocess_cityscapes_h5.py --root=../../Cityscapes/ --rescale=0.5 --dir=val --out=cscapes_val.h5
