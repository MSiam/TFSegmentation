#!/usr/bin/env bash

python preprocess_cityscapes.py --root=dummy_cityscapes/ --rescale=0.5 --output_file=174_5.h5 --mode=custom --bs=5
