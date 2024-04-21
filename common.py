import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def conv_1x1(in_channels, out_channels):
    return  nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, act=nn.ReLU(True)):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

class DynamicConvolutionModule(nn.Module):
    def __init__(self):
        super(DynamicConvolutionModule, self).__init__()
        self.feature_dim = feature_dim = 512
        self.num_kernels = num_kernels = 4
        
        # MLP for computing attention weights
        self.attention_mlp = nn.Sequential(
            nn.Linear(512, 4),
            nn.Softmax(dim=1)
        )
        
        # Trainable basis kernels and biases
        self.basis_kernels = nn.ParameterList([nn.Parameter(torch.randn(feature_dim, feature_dim)) for _ in range(num_kernels)])
        self.biases = nn.ParameterList([nn.Parameter(torch.randn(feature_dim)) for _ in range(num_kernels)])
        
    def forward(self, args):
        f_p, f_enhanced = args[0], args[1]
        # Compute attention weights
        attention_weights = self.attention_mlp(f_p)
        # Apply dynamic convolution
        sum_w = sum(attention_weights[:, :, k].unsqueeze(1) * self.basis_kernels[k] for k in range(self.num_kernels)) 
        f_o = sum_w @ f_enhanced.transpose(1,2)
        # Add dynamic biases
        sum_b = sum(attention_weights[:, :, k].unsqueeze(1) * self.biases[k] for k in range(self.num_kernels)) 
        # print(sum_b.shape, 'sum b')
        f_o = f_o.transpose(1,2) + sum_b
        
        return f_o