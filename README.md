
# Real-time Semantic Segmentation Comparative Study   
By: Mennatullah Siam, Mostafa Gamal, Moemen AbdelRazek, Senthil Yogamani, Martin Jagersand

The repository contains the official code used in the our paper [RTSEG: REAL-TIME SEMANTIC SEGMENTATION COMPARATIVE STUDY](https://arxiv.org/abs/1803.02758)

## Introduction
Semantic segmentation benefits robotics related applications especially autonomous driving. Most of the research on semantic
segmentation is only on increasing the accuracy of segmentation models with little attention to computationally efficient solutions. The few work conducted in this direction does not provide principled methods to evaluate the different design choices for segmentation. In this paper, we address this gap by presenting a real-time semantic segmentation benchmarking framework with a decoupled design for feature extraction and decoding methods. The code and the experimental results are presented on the [Cityscapes dataset for urban scenes](www.cityscapes-dataset.com).

## Feature Extractors
- VGG-16  
- ResNet-18  
- MobileNet  
- ShuffleNet

## Decoding Methods
- SkipNet   
- U-Net  
- Dilation Frontend with different subsampling factors.

## Usage
### Main Dependencies
 ```
 Python 3 and above
 tensorflow 1.3.0/1.4.0
 numpy 1.13.1
 tqdm 4.15.0
 matplotlib 2.0.2
 pillow 4.2.1
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
