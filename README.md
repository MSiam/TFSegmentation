
# Real-time Semantic Segmentation Comparative Study   
By: Mennatullah Siam, Mostafa Gamal, Moemen AbdelRazek, Senthil Yogamani, Martin Jagersand

The repository contains the official code used in the our paper [RTSEG: REAL-TIME SEMANTIC SEGMENTATION COMPARATIVE STUDY](https://arxiv.org/abs/1803.02758) for comparing different realtime semantic segmentation architectures.

## Description
Semantic segmentation benefits robotics related applications especially autonomous driving. Most of the research on semantic
segmentation is only on increasing the accuracy of segmentation models with little attention to computationally efficient solutions. The few work conducted in this direction does not provide principled methods to evaluate the different design choices for segmentation. In this paper, we address this gap by presenting a real-time semantic segmentation benchmarking framework with a decoupled design for feature extraction and decoding methods. The code and the experimental results are presented on the [CityScapes dataset for urban scenes](https://www.cityscapes-dataset.com/).

<div align="center">
<img src="https://github.com/MSiam/TFSegmentation/blob/master/figures/fig.png" width="50%" height="50%"><br><br>
</div>
## Feature Extractors
- [VGG-16](https://github.com/MSiam/TFSegmentation/blob/master/models/encoders/VGG.py)
- [ResNet-18](https://github.com/MSiam/TFSegmentation/blob/master/models/encoders/resnet_18.py)
- [MobileNet](https://github.com/MSiam/TFSegmentation/blob/master/models/encoders/mobilenet.py)
- [ShuffleNet](https://github.com/MSiam/TFSegmentation/blob/master/models/encoders/shufflenet.py)

## Decoding Methods
- SkipNet   
- U-Net  
- Dilation Frontend with different subsampling factors.

## Reported Results
### Test Set
Model | GFLOPs** | Class IoU | Class iIoU | Category IoU | Category iIoU
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
SkipNet | MobileNet | No | 61.3
SkipNet | ShuffleNet | No | 55.5
UNet | ResNet18 | No | 57.9
UNet | MobileNet | No | 61.0
UNet | ShuffleNet | No | 57.0
Dilation | MobileNet | No | 57.8
Dilation | ShuffleNet | No | 53.9
SkipNet | MobileNet | Yes | 62.4
SkipNet | ShuffleNet | Yes | 59.3

** GFLOPs is computed on image resolution 360x640. However, the mIOU(s) are computed on the official image resolution required by CityScapes evaluation script 1024x2048.

## Usage
### Run
The file named run.sh provide a good example for running different architectures. Have a look at this file.

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
