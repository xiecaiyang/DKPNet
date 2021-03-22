# -*- coding: utf-8 -*-

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26}, 
#    number={7}, 
#    pages={3142-3155}, 
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/SaoYan/DnCNN-PyTorch
# =============================================================================

# no need to run this code separately

import random
import glob
import cv2
import numpy as np
# from multiprocessing import Pool
from torch.utils.data import Dataset
import torch

patch_size, stride = 80, 20
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 64


class DenoisingDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches
        sigma: noise level, e.g., 25
    """
    def __init__(self, xs, sigma):
        super(DenoisingDataset, self).__init__()
        self.xs = xs
        self.sigma = sigma

    def __getitem__(self, index):
        #noise_type = random.randint(1,4)
        noise_type = 1 #gaussian only
        batch_x = self.xs[index]
        thre = (random.randint(1,80))/100
 
        if noise_type == 1:  #gaussian
            #rand_sigma = random.randint(0,50)
            rand_sigma = self.sigma
            noise = torch.randn(batch_x.size()).mul_(rand_sigma/255.0)
            batch_y = batch_x + noise
            thre = rand_sigma/255
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
        
        elif noise_type == 3: #impluse2
            thre_a = thre/2
            noise_gt = torch.rand(batch_x.size())
            noise = noise_gt.clone()     
            noise_image = batch_x.clone()
        
            noise_image[noise<thre_a]=0
            noise_image[(noise>thre_a)&(noise<thre)]=1
            batch_y = noise_image
        elif noise_type == 4: #poisson
            p = torch.distributions.poisson.Poisson(thre*255*batch_x)
            batch_y = p.sample()/(255*thre)
        elif noise_type == 5: #multi
            rand_sigma = random.randint(0,50)
            noise = torch.randn(batch_x.size()).mul_(rand_sigma/255.0)
            batch_y = batch_x + noise*batch_x
            thre = rand_sigma/255.0

        return batch_y, batch_x,noise_type,thre

    def __len__(self):
        return self.xs.size(0)


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def gen_patches_c(file_name):
    # get multiscale patches from a single image
    img = cv2.imread(file_name) #cv read in HxWxC(BGR) 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert image to HxWxC(RGB)
    h, w, c = img_rgb.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        img_scaled = cv2.resize(img_rgb, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                x = img_scaled[j:j+patch_size, i:i+patch_size]
                h_x, w_x, c_x = x.shape
                if(h_x!=patch_size or w_x!=patch_size):
                    print(h_scaled, w_scaled, i, j)
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)
    return patches


def datagenerator_c(data_dir='data/Train400', verbose=False):
    # generate clean patches from a dataset
    file_list = glob.glob(data_dir+'/*.jpg') + glob.glob(data_dir+'/*.png') # get name list of all .png files
    # initialize
    data = []
    # generate patches
    for i in range(len(file_list)):
        patches = gen_patches_c(file_list[i])
        for patch in patches:    
            data.append(patch)
        if verbose:
            print(str(i+1) + '/' + str(len(file_list)) + ' is done ^_^')
    data = np.array(data, dtype='uint8')
    #data = np.expand_dims(data, axis=3)
    #discard_n = len(data)-len(data)//batch_size*batch_size  # because of batch namalization
    #data = np.delete(data, range(discard_n), axis=0)
    #print("discard {} patches for batch normalization!".format(discard_n))
    print('^_^-training data finished-^_^')
    return data

def gen_patches(file_name):
    # get multiscale patches from a single image
    img = cv2.imread(file_name, 0)  # imread_flag 0 = gray scales
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                x = img_scaled[j:j+patch_size, i:i+patch_size]
                h_x, w_x = x.shape
                if(h_x!=patch_size or w_x!=patch_size):
                    print(h_scaled, w_scaled, i, j)
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)
    return patches


def datagenerator(data_dir='data/Train400', verbose=False):
    # generate clean patches from a dataset
    file_list = glob.glob(data_dir+'/*.jpg') + glob.glob(data_dir+'/*.png')  # get name list of all .png files
    # initialize
    data = []
    # generate patches
    for i in range(len(file_list)):
        patches = gen_patches(file_list[i])
        for patch in patches:    
            data.append(patch)
        if verbose:
            print(str(i+1) + '/' + str(len(file_list)) + ' is done ^_^')
    data = np.array(data, dtype='uint8')
    data = np.expand_dims(data, axis=3)
    #discard_n = len(data)-len(data)//batch_size*batch_size  # because of batch namalization
    #data = np.delete(data, range(discard_n), axis=0)
    #print("discard {} patches for batch normalization!".format(discard_n))
    print('^_^-training data finished-^_^')
    return data

if __name__ == '__main__': 

    data = datagenerator(data_dir='data/Train400')


#    print('Shape of result = ' + str(res.shape))
#    print('Saving data...')
#    if not os.path.exists(save_dir):
#            os.mkdir(save_dir)
#    np.save(save_dir+'clean_patches.npy', res)
#    print('Done.')       
