import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.init import kaiming_normal_
from torch.autograd import Variable
from torch.autograd import grad
import torch.autograd as autograd

import numpy as np
from collections import OrderedDict

import os
import glob

import cv2 as cv


class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight.view(1, -1, 1, 1) * noise.to(x.device)


class FC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 gain=2**(0.5),
                 use_wscale=False,
                 lrmul=1.0,
                 bias=True):
        """
            The complete conversion of Dense/FC/Linear Layer of original Tensorflow version.
        """

        super(FC, self).__init__()
        he_std = gain * in_channels ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            out = F.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul)
        else:
            out = F.linear(x, self.weight * self.w_lrmul)
        out = F.leaky_relu(out, 0.2, inplace=True)
        return out



class ApplyStyle(nn.Module):
    def __init__(self, latent_size, channels, use_wscale):
        super(ApplyStyle, self).__init__()
        self.linear = FC(latent_size,
                      channels * 2,
                      gain=1.0,
                      use_wscale=use_wscale)

    # x: (M, c, X, X) <> latent: w(M, 128)
    def forward(self, x, latent):
        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        # M x 2*c
        shape = [-1, 2, x.size(1), 1, 1]
        # M x 2 x c x 1 x 1
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        # a, b
        # x = x * (a+1) + b
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x

class Blur2d(nn.Module):
    def __init__(self, f=[1,2,1], normalize=True, flip=False, stride=1):
        """
            depthwise_conv2d:
            https://blog.csdn.net/mao_xiao_feng/article/details/78003476
        """
        super(Blur2d, self).__init__()
        assert isinstance(f, list) or f is None, "kernel f must be an instance of python built_in type list!"

        if f is not None:
            f = torch.tensor(f, dtype=torch.float32)
            f = f[:, None] * f[None, :]
            f = f[None, None]
            if normalize:
                f = f / f.sum()
            if flip:
                # f = f[:, :, ::-1, ::-1]
                f = torch.flip(f, [2, 3])
            self.f = f
        else:
            self.f = None
        self.stride = stride

    def forward(self, x):
        if self.f is not None:
            # expand kernel channels
            kernel = self.f.expand(x.size(1), -1, -1, -1).to(x.device)
            x = F.conv2d(
                x,
                kernel,
                stride=self.stride,
                padding=int((self.f.size(2)-1)/2),
                groups=x.size(1)
            )
            return x
        else:
            return x


class Conv2d(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 gain=2 ** (0.5),
                 use_wscale=False,
                 lrmul=1,
                 bias=True):
        super().__init__()
        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)  # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            return F.conv2d(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul, padding=self.kernel_size // 2)
        else:
            return F.conv2d(x, self.weight * self.w_lrmul, padding=self.kernel_size // 2)


class Upscale2d(nn.Module):
    def __init__(self, factor=2, gain=1):
        super().__init__()
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        if self.gain != 1:
            x = x * self.gain
        if self.factor > 1:
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, self.factor, -1, self.factor)
            x = x.contiguous().view(shape[0], shape[1], self.factor * shape[2], self.factor * shape[3])
        return x



class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp  = torch.mul(x, x) # or x ** 2 [M, c, X, X]
        tmp1 = torch.rsqrt(torch.mean(tmp, dim=1, keepdim=True) + self.epsilon)

        return x * tmp1


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x) # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


class LayerEpilogue(nn.Module):
    def __init__(self,
                 channels,
                 dlatent_size,
                 use_wscale,
                 use_noise,
                 use_pixel_norm,
                 use_instance_norm,
                 use_styles):
        super(LayerEpilogue, self).__init__()

        if use_noise:
            self.noise = ApplyNoise(channels)
        self.act = nn.LeakyReLU(negative_slope=0.2)

        if use_pixel_norm:
            self.pixel_norm = PixelNorm()
        else:
            self.pixel_norm = None

        if use_instance_norm:
            self.instance_norm = InstanceNorm()
        else:
            self.instance_norm = None

        if use_styles:
            self.style_mod = ApplyStyle(dlatent_size, channels, use_wscale=use_wscale)
        else:
            self.style_mod = None

    def forward(self, x, noise, dlatents_in_slice=None):
        x = self.noise(x, noise)
        x = self.act(x)
        if self.pixel_norm is not None:
            x = self.pixel_norm(x)
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        if self.style_mod is not None:
            x = self.style_mod(x, dlatents_in_slice)

        return x

class GBlock(nn.Module):
    def __init__(self,
                 res,
                 use_wscale,
                 use_noise,
                 use_pixel_norm,
                 use_instance_norm,
                 noise_input,        # noise
                 dlatent_size=128,   # Disentangled latent (W) dimensionality.
                 use_style=True,     # Enable style inputs?
                 f=None,        # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 factor=2,           # upsample factor.
                 fmap_base=2048,     # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,     # log2 feature map reduction when doubling the resolution.
                 fmap_max=128,       # Maximum number of feature maps in any layer.
                 ):
        super(GBlock, self).__init__()
        # self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        #           0,   1,   2,   3 ,  4 ,  5 ,  6 ,  7 
        self.nf = [128, 128, 128, 128, 128, 128, 64, 32 ]
        # res
        self.res = res

        # blur2d
        self.blur = Blur2d(f)

        # noise
        self.noise_input = noise_input

        if res < 6:
            # upsample method 1
            self.up_sample = Upscale2d(factor)
        else:
            self.up_sample = nn.ConvTranspose2d(self.nf[res-1], self.nf[res], 4, stride=2, padding=1)

        # A Composition of LayerEpilogue and Conv2d.
        self.adaIn1 = LayerEpilogue(self.nf[res], dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style)
        self.conv1  = Conv2d(input_channels=self.nf[res], output_channels=self.nf[res],
                             kernel_size=3, use_wscale=use_wscale)
        self.adaIn2 = LayerEpilogue(self.nf[res], dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style)

    def forward(self, x, dlatent):
        x = self.up_sample(x)
        x = self.adaIn1(x, self.noise_input[self.res*2-4], dlatent[:, self.res*2-4])
        x = self.conv1(x)
        x = self.adaIn2(x, self.noise_input[self.res*2-3], dlatent[:, self.res*2-3])
        return x