#!/usr/bin/env bash

# Overfit test run
python main.py --load_config=test.yaml overfit Train FCN8s

# FCN8s exp 1 run
#python main.py --load_config=fcn8s_exp_1.yaml train Train FCN8s
