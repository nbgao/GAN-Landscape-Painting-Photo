import torch
import torch.nn as nn
from torch.nn.parameter import parameter

class ResnetGenerator(nn.Module):
    def __init__():

    
    def forward(self, input):


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias):


    def forward(self, x):


class ResnetAdaILNBlock(nn.Module):
    def __init__(self, dim, use_bias):


    def forward(self, x):


class AdaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):

    def forward(self, input, gamma, beta):


class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):

    def forward(self, input):

class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5):


    def forward(self, input):


class RhoClipper(object):
    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max
    
    def __call__(self, module):
        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self, clip_min, clip_max)
            module.rho.data = w