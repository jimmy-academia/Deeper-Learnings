# -*- coding: utf-8 -*-
"""Function for DCGAN model class delaration
__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class model_param():
    def __init__(self):
        self.img_channel_num = 1
        self.z_dim = 100
        self.ngf = 128
        self.ndf = 128
        self.layer_G = [(self.ngf*8,4,1,0), (self.ngf*4,4,2,1), (self.ngf*2,4,2,1), (self.ngf,4,2,1), (self.img_channel_num,4,2,1)]
        self.layer_D = [(self.ndf,4,2,1), (self.ndf*2,4,2,1), (self.ndf*4,4,2,1), (self.ndf*8,4,2,1), (1,4,1,0)]

class DCGAN(nn.Module):
    def __init__(self, args=None):
        super(DCGAN, self).__init__()
        if args is None:
            args = model_param()
        self.G = Generator(args)
        self.G.apply(weights_init)
        self.D = Discriminator(args)
        self.D.apply(weights_init)

    def save(self, filepath):
        state = {
            'gen_net': self.G.state_dict(),
            'dis_net': self.D.state_dict(),
        }
        torch.save(state, filepath)

    def load(self, filepath):
        state = torch.load(filepath)
        self.G.load_state_dict(state['gen_net'])
        self.D.load_state_dict(state['dis_net'])

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(m.bias, 0.0)

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.main = nn.ModuleList()
        in_channel = args.z_dim
        for i, x in enumerate(args.layer_G):
            self.main.append(nn.ConvTranspose2d(in_channel, *x))
            in_channel = x[0]
            if i < len(args.layer_G)-1:
                self.main.append(nn.BatchNorm2d(in_channel))
                self.main.append(nn.ReLU())
            else:
                self.main.append(nn.Tanh())

    def forward(self,x):
        for layer in self.main:
            x = layer(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.main = nn.ModuleList()
        in_channel = args.img_channel_num
        for i, x in enumerate(args.layer_D):
            self.main.append(nn.Conv2d(in_channel, *x))
            in_channel = x[0]
            if i > 0 and i < len(args.layer_D)-1:
                self.main.append(nn.BatchNorm2d(in_channel))
                
            if i < len(args.layer_D)-1:
                self.main.append(nn.LeakyReLU(0.2))
            else:
                self.main.append(nn.Sigmoid())
            
    def forward(self,x):
        for layer in self.main:
            x = layer(x)
        return x


