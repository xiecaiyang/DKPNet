import torch
import collections
import random 
import math
from PIL import Image
import numpy as np
import numbers
import torchvision.transforms as transforms
#from torchvision.transforms.functional import resized_crop,\
#resize, hflip, to_tensor, crop
'''
for train
    size: rescale image size 
    mean: sequence of means for R,G,B channels
    std: sequence of standard deviations for R,G,B channels
'''
to_tensor = transforms.ToTensor()
class AddGaussianNoise(object):
    def __init__(self, mean, sigma):
        self.sigma = sigma
        self.mean = mean
    def __call__(self, image):
        ch, row, col = image.size()
        gauss = torch.Tensor(ch, row, col)
        sigma = self.sigma  / 255.0
        gauss.normal_(self.mean, sigma)
        noise_image = gauss + image
        return noise_image
        #return image

class SingleRandomCropTransform(object):
    def __init__(self, size, noise_level, interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.noise_level = noise_level
    def __call__(self, image):
        #transforms.CenterCrop(32)
        crop = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])
        tensor_image = crop(image);
        #tmp_noise_level = random.randint(1,55)
        noise_image = AddGaussianNoise(0, self.noise_level)(tensor_image)
        return noise_image, tensor_image
'''
for single crop test
'''
class SingleTransform(object):
    def __init__(self, noise_level, interpolation=Image.BILINEAR,noise_type = 1):
        self.interpolation = interpolation
        self.noise_level = noise_level
        self.noise_type = noise_type

    def __call__(self, image):
        batch_x = to_tensor(image)
        noise_type = self.noise_type
        thre = self.noise_level/100
        thre_a = thre
        if noise_type == 1:  #gaussian
            #rand_sigma = 25#random.randint(0,50)
            rand_sigma = self.noise_level
            noise = torch.randn(batch_x.size()).mul_(rand_sigma/255.0)
            batch_y = batch_x + noise
            thre_a = rand_sigma/255.0
        elif noise_type == 2:  #impluse1                 
            noise_gt = torch.rand(batch_x.size())
            noise = noise_gt.clone()     
            noise_image = batch_x.clone()
            noise[noise>=thre] = 0
            noise_tmp = noise[noise>0]
            lmn = torch.min(noise_tmp)
            lmax = torch.max(noise_tmp)
            noise = ((noise-lmn))/(lmax-lmn)
            mask = torch.ceil(noise_gt-thre)
            noise_image = noise*(1-mask)+noise_image*mask
            batch_y = noise_image
        
        elif noise_type == 3: #impluse2(salt-and-pepper)
            thre = thre/2
            noise_gt = torch.rand(batch_x.size())
            noise = noise_gt.clone()     
            noise_image = batch_x.clone()
        
            noise_image[noise<thre]=0
            noise_image[(noise>thre)&(noise<2*thre)]=1
            batch_y = noise_image
        elif noise_type == 4: #poisson
            p = torch.distributions.poisson.Poisson(thre*255*batch_x)
            batch_y = p.sample()/(255*thre)
        elif noise_type == 5: #multi
            rand_sigma = 25#random.randint(0,50)
            noise = torch.randn(batch_x.size()).mul_(rand_sigma/255.0)
            batch_y = batch_x + noise*batch_x
            thre_a = rand_sigma/255.0
           

        return batch_y, batch_x,noise_type,thre_a

