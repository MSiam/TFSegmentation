# Real-time Semantic Segmentation Comparative Study
By: [Mennatullah Siam](https://github.com/MSiam), [Mostafa Gamal](https://github.com/MG2033), [Moemen AbdelRazek](https://github.com/moemen95), Senthil Yogamani, Martin Jagersand

The repository contains the official **TensorFlow** code used in the our paper [RTSEG: REAL-TIME SEMANTIC SEGMENTATION COMPARATIVE STUDY](https://arxiv.org/abs/1803.02758) for comparing different realtime semantic segmentation     architectures.

## Description
Semantic segmentation benefits robotics related applications especially autonomous driving. Most of the research on semantic segmentation is only on increasing the accuracy of segmentation models with little attention to computationally efficient solutions. The few work conducted in this direction does not provide principled methods to evaluate the      different design choices for segmentation. In this paper, we address this gap by presenting a real-time semantic segmentation benchmarking framework with a decoupled design for feature extraction and decoding methods. The code and the experimental results are presented on the [CityScapes dataset for urban scenes](https://www.cityscapes-dataset.com/).

<div align="center">
<img src="https://github.com/MSiam/TFSegmentation/blob/master/figures/fig.png" width="70%" height="70%"><br><br>
</div>

## Models
Encoder | Skip | U-Net | DilationV1 | DilationV2
------- | ---- | ----- | ---------- | ----------
[VGG-16](https://github.com/MSiam/TFSegmentation/blob/master/models/encoders/VGG.py) | [Yes](https://github.com/MSiam/TFSegmentation/blob/master/models/fcn8s.py) | [Yes](https://github.com/MSiam/TFSegmentation/blob/master/models/unet_vgg16.py) | [Yes](https://github.com/MSiam/TFSegmentation/blob/master/models/dilation.py) | No
[ResNet-18](https://github.com/MSiam/TFSegmentation/blob/master/models/encoders/resnet_18.py) | [Yes](https://github.com/MSiam/TFSegmentation/blob/master/models/fcn8s_resnet18.py) | [Yes](https://github.com/MSiam/TFSegmentation/blob/master/models/linknet.py) | [Yes](https://github.com/MSiam/TFSegmentation/blob/master/models/dilation_resnet18.py) | No
[MobileNet](https://github.com/MSiam/TFSegmentation/blob/master/models/encoders/mobilenet.py) | [Yes](https://github.com/MSiam/TFSegmentation/blob/master/models/fcn8s_mobilenet.py) | [Yes](https://github.com/MSiam/TFSegmentation/blob/master/models/unet_mobilenet.py) | [Yes](https://github.com/MSiam/TFSegmentation/blob/master/models/dilation_mobilenet.py) | [Yes](https://github.com/MSiam/TFSegmentation/blob/master/models/dilationv2_mobilenet.py)
[ShuffleNet](https://github.com/MSiam/TFSegmentation/blob/master/models/encoders/shufflenet.py) | [Yes](https://github.com/MSiam/TFSegmentation/blob/master/models/fcn8s_shufflenet.py) | [Yes](https://github.com/MSiam/TFSegmentation/blob/master/models/unet_shufflenet.py) | [Yes](https://github.com/MSiam/TFSegmentation/blob/master/models/dilation_shufflenet.py) | [Yes](https://github.com/MSiam/TFSegmentation/blob/master/models/dilationv2_shufflenet.py)

**NOTE: The rest of the pretrained weights for all the implemented models will be released soon. Stay in touch for the updates.**

## Reported Results
### Test Set
Model | GFLOPs | Class IoU | Class iIoU | Category IoU | Category iIoU
----- | ------ | --------- | ---------- | ------------ | -------------
SegNet | 286.03 | 56.1 | 34.2 | 79.8 | 66.4
ENet | 3.83 | 58.3 | 24.4 | 80.4 | 64.0
DeepLab | - | 70.4 | 42.6 | 86.4 | 67.7
SkipNet-VGG16 | - | 65.3 | 41.7 | 85.7 | 70.1
SkipNet-ShuffleNet | 2.0 | 58.3 | 32.4 | 80.2 | 62.2
SkipNet-MobileNet | 6.2 | 61.5 | 35.2 | 82.0 | 63.0

### Validation Set
Encoder | Decoder | Coarse | mIoU
------- | ------- | ------ | ----
MobileNet | SkipNet | No | 61.3
ShuffleNet | SkipNet | No | 55.5
ResNet-18 | UNet | No | 57.9
MobileNet | UNet | No | 61.0
ShuffleNet | UNet | No | 57.0
MobileNet | Dilation | No | 57.8
ShuffleNet | Dilation | No | 53.9
MobileNet | SkipNet | Yes | 62.4
ShuffleNet | SkipNet | Yes | 59.3

** GFLOPs is computed on image resolution 360x640. However, the mIOU(s) are computed on the official image resolution required by CityScapes evaluation script 1024x2048.**

## Usage
1. Download the weights, processed data, and trained weights from [here](https://drive.google.com/drive/folders/19lJhjiYTKIBrCPwi0cxn1xuyLE3d9i6O?usp=sharing)
2. Extract pretrained_weights.zip
3. Extract full_cityscapes_res.zip under data/
4. Extract unet_resnet18.zip under experiments/

### Run
The file named run.sh provide a good example for running different architectures. Have a look at this file.
1. Remove comment from run.sh for running fcn8s_mobilenet on the validation set of cityscapes to get its mIoU.
Our framework evaluation will produce results lower than the cityscapes evaluation script by small difference, for the final evaluation we use the cityscapes evaluation script. UNet ResNet18 should have 56% on validation set, but with cityscapes script we got 57.9%. The results on the test set for SkipNet-MobileNet and SkipNet-Shufflenet are publicly available on the Cityscapes Benchmark.
```
python3 main.py --load_config=unet_resnet18_test.yaml test Train LinkNET
```
2. To measure running time run in inference mode.
```
python3 main.py --load_config=unet_resnet18_test.yaml inference Train LinkNET
```
3. To run on different dataset modify the configuration file config/experiments_config/unet_resnet18_test.yaml

**NOTE: The current code does not contain the optimized code for measuring inference time, the final code will be released soon.**

#### Example to the running command written in the file:
```
python3 main.py --load_config=[config_file_name].yaml [train/test] [Trainer Class Name] [Model Class Name]
```
### Main Dependencies
 ```
 Python 3 and above
 tensorflow 1.3.0/1.4.0
 numpy 1.13.1
 tqdm 4.15.0
 matplotlib 2.0.2
 pillow 4.2.1
 PyYAML 3.12
 ```
### All Dependencies
 ```
 pip install -r [requirements_gpu.txt] or [requirements.txt]
 ```

## Citation
If you find RTSeg useful in your research, please consider citing our work:

```
@ARTICLE{2018arXiv180302758S,
   author = {{Siam}, M. and {Gamal}, M. and {Abdel-Razek}, M. and {Yogamani}, S. and
    {Jagersand}, M.},
    title = "{RTSeg: Real-time Semantic Segmentation Comparative Study}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1803.02758},
 primaryClass = "cs.CV",
 keywords = {Computer Science - Computer Vision and Pattern Recognition},
     year = 2018,
    month = mar,
   adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180302758S},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.


