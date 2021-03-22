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

# DKPNet9: dilation implemented in Conv2d in the feature branches instead of the kernel conv
#          two dilated conv layers in ResDltBlock
#          kernel core weighted addition

import argparse
import os, time, datetime
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
import torch.nn.functional as F
from math import sqrt
from tensorboardX import SummaryWriter

import common

class KernelConv(nn.Module):
    """
    the class of computing kernel prediction
    """
    def __init__(self, kp_size=5):
        super(KernelConv, self).__init__()
        self.kernel_size = kp_size

    def forward(self, image, core, dilation):
        """
        compute the pred image according to kernel(core) and feature map(image) 
        :param image: [batch_size, 1, height, width]
        :param core: [batch_size, kxk, (1), height, width] #channel=1,thus squeezed before input
        :return:
        """
        batch_size, c, h, w = image.size()
        #print("b,c,h,w={},{},{},{}".format(batch_size, c, h, w))
        K = self.kernel_size
        stride = dilation
        pad_size = (K//2)*stride
        core = core.view(batch_size, K*K, 1, h, w)

        img_stack = []
        pred_img = []
        frame_pad = F.pad(image, [pad_size, pad_size, pad_size, pad_size])
        #print("frame_pad.size={}".format(frame_pad.size()))

        # Stack shifted images to align the K*K kernel region
        for i in range(K):
            for j in range(K):
                iii = frame_pad[..., i*stride:i*stride+h, j*stride:j*stride+w]
                img_stack.append(iii)
                #print("iii.size={}".format(iii.size()))
        img_stack = torch.stack(img_stack, dim=1)

        # Core:(batch_size, K*K, 1, h, w)  img_stack:(batch_size, K*K, n_feats, h, w)
        # First sum over dim(K*K) to get per-pixel conv in every feature map
        # Then sum over dim(n_feats) to squeeze all feature maps to one output channel
        pred_img = torch.sum(
                         torch.sum(core.mul(img_stack), dim=1, keepdim=False),
                        dim=1, keepdim=True)
        return pred_img

class ResDltBlock(nn.Module):
    def __init__(
        self, dilation, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResDltBlock, self).__init__()
        padding=(kernel_size//2)
        m = []
        m.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(padding*dilation), dilation=dilation, bias=bias))
        m.append(act)
        m.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(padding*dilation), dilation=dilation, bias=bias))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class DKPNet(nn.Module):    
    def __init__(self, n1=9, n2=3, n_feats=64, kp_size=3,  conv=common.default_conv):
        super(DKPNet, self).__init__()
        padding = 1
        img_channels = 1
        kernel_size = 3
        n1_resblocks = n1
        n2_resblocks = n2
        res_scale = 1
        act = nn.ReLU(True)
        self.kernel_pred = KernelConv(kp_size)
        
        head = [conv(img_channels, n_feats, kernel_size)]
        # define the structure of shared feature extraction
        m_sfe = [common.ResBlock(
                    conv, n_feats, kernel_size, act=act, res_scale=res_scale
                 ) for _ in range(n1_resblocks)]
        m_sfe.append(conv(n_feats, n_feats, kernel_size))
        
        # define the structure of branch feature extraction 
        # d1,d2,d3 branches share the same structure, but of individual entities 
        m_b1fe = [ResDltBlock(
                    1, n_feats, kernel_size, act=act, res_scale=res_scale
                  ) for _ in range(n2_resblocks)]
        m_b1fe.append(conv(n_feats, n_feats, kernel_size))
        m_b2fe = [ResDltBlock(
                    2, n_feats, kernel_size, act=act, res_scale=res_scale
                  ) for _ in range(n2_resblocks)]
        m_b2fe.append(conv(n_feats, n_feats, kernel_size))
        m_b3fe = [ResDltBlock(
                    3, n_feats, kernel_size, act=act, res_scale=res_scale
                  ) for _ in range(n2_resblocks)]
        m_b3fe.append(conv(n_feats, n_feats, kernel_size))

        # The main pipeline of the network
        self.head = nn.Sequential(*head)
        self.sfe = nn.Sequential(*m_sfe)
        # Compressed feature map
        self.cmprss = conv(n_feats, 1, kernel_size)
        #self.cmprss = DeformConv2d(n_feats, 1, kernel_size, padding=1, bias=False, modulation=True) 
        # Feature branches
        self.d1fe = nn.Sequential(*m_b1fe)
        self.d2fe = nn.Sequential(*m_b2fe)
        self.d3fe = nn.Sequential(*m_b3fe)
        # generate kernel-prediction filter of dilations 1,2,3
        self.d1kp = conv(n_feats, kp_size*kp_size, kernel_size)
        self.d2kp = conv(n_feats, kp_size*kp_size, kernel_size)
        self.d3kp = conv(n_feats, kp_size*kp_size, kernel_size)
        # kernel core fusion
        self.fusion = conv(3*kp_size*kp_size, kp_size*kp_size, kernel_size)
        # 3x3 reconstrution convolution, channel 1->1
        self.rec = conv(1, 1, kernel_size)

    def forward(self, origin_x):
        x = origin_x.clone()
        
        x_t = self.head(x)
        # Shared feature extraction
        x_fms = x_t + self.sfe(x_t)

        # Compressed feature map
        x_f = self.cmprss(x_fms)

        # d1 branch
        kpd1 = self.d1kp(x_fms + self.d1fe(x_fms))
        # d2 branch
        kpd2 = self.d2kp(x_fms + self.d2fe(x_fms))
        # d3 branch
        kpd3 = self.d3kp(x_fms + self.d3fe(x_fms))

        # kernel core fusion
        kpAll = torch.cat((kpd1, kpd2, kpd3), 1)
        kpf = self.fusion(kpAll)
        #Per-pixel kernel convolution
        kpRi = self.kernel_pred(x_f, kpf, 1)

        out_img = self.rec(kpRi)

        return out_img + origin_x

