# DKPNet
Code for 'Dilated kernel prediction network for single-image denoising'

Done in environment: Python 3.6.8, Pytorch 1.1.0, cuda 10.0, cudnn 7.6.5. Supports gray-image denoising. Color-image denoising is still in progress.

Get started by simply running the train/eval script! 

The training was carried out on BSD500 in our experiments. Remember to change the dataset directory according to your own environment and make sure the network setting is consistent with the pretrained model (kp_size=3, n1=9, n2=2).
