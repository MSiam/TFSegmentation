
# RTSeg
RTSeg: Real-time Semantic Segmentation Comparative Study   
By: Mennatullah Siam, Mostafa Gamal, Moemen AbdelRazek, Senthil Yogamani, Martin Jagersand

The repository contains the official Code used in the comparative study between different design choices for real-time semantic segmentation.

Citing RTSeg: 
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

# Encoders
ResNet-18  
VGG-16  
MobileNet  
ShuffleNet

# Deconding Methods
SkipNet   
U-Net  
Dilation

# Dependencies
Python 3.5.2  
Tensorflow 1.4

1- pip install -r requirements_gpu.txt   
2- ./run.sh
