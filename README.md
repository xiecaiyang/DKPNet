# DKPNet
Code for paper 'Dilated kernel prediction network for single-image denoising'


GET STARTED BY CHECKING OUT THE TRAIN/EVAL SCRIPT!


Pre-trained model files are provided in /checkpoint/pretrained_models. All three models here are of the standard setting (kp_size=3, n1=9, n2=2), but of different noise levels.





NOTICE: 

Our work was done in the following environment: Python 3.6.8, Pytorch 1.1.0, cuda 10.0, cudnn 7.6.5.

We used BSD500 as our training and validation datasets. Remember to change the directory in the script according to your own dataset location. While evaluating, make sure the network setting is consistent with the selected model file.

The training/evaluation of gray-image denoising has been tested. Color-image denoising is still in progress.
