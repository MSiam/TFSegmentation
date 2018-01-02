#!/usr/bin/env bash

######################################## VGG16 ######################################################
#1- FCN8s VGG16 Train Coarse+Fine
#python3 main.py --load_config=fcn8s_vgg16_traincoarse.yaml train Train FCN8s
#python3 main.py --load_config=fcn8s_vgg16_train.yaml train Train FCN8s

#2- FCN8s VGG16 Test
#python3 main.py --load_config=fcn8s_vgg16_test.yaml test Train FCN8s
#python3 main.py --load_config=fcn8s_vgg16_test.yaml inference Train FCN8s

#3- UNET VGG16 Train
#python3 main.py --load_config=unet_vgg16_train.yaml train Train VGG16UNET

#4- UNET VGG16 Test
#python3 main.py --load_config=unet_vgg16_test.yaml test Train VGG16UNET

#5- Dilation v1 VGG16 Train

#6- Dilation v1 VGG16 Test

###################################### MobileNet ##################################################
#1- FCN8s MobileNet Train Coarse+Fine
#python3 main.py --load_config=fcn8s_mobilenet_traincoarse.yaml train Train FCN8sMobileNet
#python3 main.py --load_config=fcn8s_mobilenet_train.yaml train Train FCN8sMobileNet

#2- FCN8s MobileNet Test
#python3 main.py --load_config=fcn8s_mobilenet_test.yaml test Train FCN8sMobileNet

#3- UNet MobileNet Train Coarse+Fine
#python3 main.py --load_config=unet_mobilenet_traincoarse.yaml train Train UNetMobileNet
#python3 main.py --load_config=unet_mobilenet_train.yaml train Train UNetMobileNet

#4- UNet MobileNet Test 
#python3 main.py --load_config=unet_mobilenet_test.yaml test Train UNetMobileNet

#5- Dilation v1 MobileNet Train
#python3 main.py --load_config=dilation_mobilenet_train.yaml train Train DilationMobileNet

#6- Dilation v1 MobileNet Test

#7- Dilation v2 MobileNet Train 
python3 main.py --load_config=dilationv2_mobilenet_train.yaml train Train DilationV2MobileNet

###################################### ShuffleNet #################################################
#1- FCN8s ShuffleNet Train Coarse+Fine
#python3 main.py --load_config=fcn8s_shufflenet_traincoarse.yaml train Train FCN8sShuffleNet
#python3 main.py --load_config=fcn8s_shufflenet_train.yaml train Train FCN8sShuffleNet

#2- FCN8s ShuffleNet Test
#python3 main.py --load_config=fcn8s_shufflenet_test.yaml test Train FCN8sShuffleNet
#python3 main.py --load_config=fcn8s_shufflenet_test.yaml inference Train FCN8sShuffleNet

#3- UNet ShuffleNet Train Coarse+Fine
#python3 main.py --load_config=fcn8s_shufflenet_traincoarse.yaml train Train FCN8sShuffleNet
#python3 main.py --load_config=fcn8s_shufflenet_train.yaml train Train FCN8sShuffleNet

#4- UNet ShuffleNet Test
#python3 main.py --load_config=fcn8s_shufflenet_test.yaml test Train FCN8sShuffleNet

#5- Dilation v1 ShuffleNet Train
#python3 main.py --load_config=dilation_shufflenet_train.yaml train Train DilationShuffleNet

##################################### ResNet18 ####################################################
#3- UNet ResNet18 Train
#python3 main.py --load_config=unet_resnet18_traincoarse.yaml train Train LinkNET
#python3 main.py --load_config=unet_resnet18_train.yaml train Train LinkNET

# - UNet ResNet18 Inference .. FPS Test
#python3 main.py --load_config=unet_resnet18_test.yaml inference_pkl Train LinkNET
#python3 main.py --load_config=unet_resnet18_test.yaml debug Train LinkNET

#4- UNet ResNet18 Test
#python3 main.py --load_config=unet_resnet18_test.yaml test Train LinkNET
#python3 main.py --load_config=unet_resnet18_test.yaml inference Train LinkNET

#5- Dilation v1 ResNet18 Train
#python3 main.py --load_config=dilation_resnet18_train.yaml train Train DilationResNet18

#6- Dilation v1 ResNet18 Test
