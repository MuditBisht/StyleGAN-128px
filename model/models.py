import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

from .utils import (
    ApplyNoise, FC, ApplyStyle, Blur2d, Conv2d, Upscale2d, 
    PixelNorm, InstanceNorm, LayerEpilogue,GBlock)


class G_mapping(nn.Module):
    def __init__(self,
                    mapping_fmaps=128,
                    dlatent_size=128,
                    resolution=128,
                    normalize_latents=True,  # Normalize latent vectors (Z) before feeding them to the mapping layers?
                    use_wscale=True,         # Enable equalized learning rate?
                    lrmul=0.01,              # Learning rate multiplier for the mapping layers.
                    gain=2**(0.5)            # original gain in tensorflow.
        ):
        super(G_mapping, self).__init__()
        self.mapping_fmaps = mapping_fmaps
        
        self.func = nn.Sequential(
            FC( self.mapping_fmaps, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale ),
            FC( dlatent_size,       dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale ),
            FC( dlatent_size,       dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale ),
            FC( dlatent_size,       dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale ),
            FC( dlatent_size,       dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale ),
            FC( dlatent_size,       dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale ),
            FC( dlatent_size,       dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale ),
            FC( dlatent_size,       dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale )
        )

        self.normalize_latents = normalize_latents
        self.resolution_log2 = int(np.log2(resolution)) # 7
        self.num_layers = self.resolution_log2 * 2 - 2 # 12
        self.pixel_norm = PixelNorm()
        # - 2 means we start from feature map with height and width equals 4.
        # as this example, we get num_layers = 18.

    def forward(self, x):
        if self.normalize_latents:
            x = self.pixel_norm(x)
        out = self.func(x)
        return out, self.num_layers # Tensor(M, 128), 12


class G_synthesis(nn.Module):
    def __init__(self,
                 dlatent_size=128,                       # Disentangled latent (W) dimensionality.
                 resolution=128,                    # Output resolution (1024 x 1024 by default).
                 fmap_base=512,                     # Overall multiplier for the number of feature maps.
                 num_channels=3,                     # Number of output color channels.
                 structure='fixed',                  # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
                 fmap_max=128,                       # Maximum number of feature maps in any layer.
                 fmap_decay=1.0,                     # log2 feature map reduction when doubling the resolution.
                 f=None,                             # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 use_pixel_norm      = False,        # Enable pixelwise feature vector normalization?
                 use_instance_norm   = True,         # Enable instance normalization?
                 use_wscale = True,                  # Enable equalized learning rate?
                 use_noise = True,                   # Enable noise inputs?
                 use_style = True,                    # Enable style inputs?
                 device="cuda"
    ):                             # batch size.
        super(G_synthesis, self).__init__()

        self.structure = structure

        num_layers = 12
        self.num_layers = num_layers

        # Noise inputs.
        self.noise_inputs = []
        for layer_idx in range(num_layers):
          res = layer_idx // 2 + 2
          shape = [1, 1, min(resolution, 2 ** res), min(resolution, 2 ** res)]  # M * c * X * X
          self.noise_inputs.append(torch.randn(*shape).to(device))

        # Blur2d
        self.blur = Blur2d(f)

        # torgb: fixed mode
        self.channel_shrinkage = Conv2d(input_channels=32,
                                        output_channels=8,
                                        kernel_size=3,
                                        use_wscale=use_wscale)
        self.torgb = Conv2d(8, num_channels, kernel_size=1, gain=1, use_wscale=use_wscale)

        # Initial Input Block
        self.const_input = nn.Parameter(torch.ones(1, 128, 4, 4))
        self.bias = nn.Parameter(torch.ones(128))
        
        self.adaIn1 = LayerEpilogue(128, 128, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style)
        self.conv1  = Conv2d(input_channels=128, output_channels=128, kernel_size=3, use_wscale=use_wscale)
        
        self.adaIn2 = LayerEpilogue(128, 128, use_wscale, use_noise, use_pixel_norm,
                                    use_instance_norm, use_style)

        # Common Block
        # 128 x 4 x 4 -> 128 x 4 x 4
        res = 3
        self.GBlock1 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 128 x 8 x 8 -> 128 x 16 x 16
        res = 4
        self.GBlock2 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 128 x 16 x 16 -> 128 x 32 x 32
        res = 5
        self.GBlock3 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 128 x 32 x 32 -> 64 x 64 x 64
        res = 6
        self.GBlock4 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 64 x 64 x 64 -> 32 x 128 x 128
        res = 7
        self.GBlock5 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # # 128 x 128 x 128 -> 64 x 128 x 128
        # res = 8
        # self.GBlock6 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
        #                       self.noise_inputs)

        # # 64 x 128 x 128 -> 32 x 128 x 128
        # res = 9
        # self.GBlock7 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
        #                       self.noise_inputs)

        # # 512 x 512 -> 1024 x 1024
        # res = 10
        # self.GBlock8 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
        #                       self.noise_inputs)

    def forward(self, dlatent):
        """
           dlatent: Disentangled latents (W), shape为[minibatch, num_layers, dlatent_size].
        :param dlatent:
        :return:
        """
        images_out = None
        # Fixed structure: simple and efficient, but does not support progressive growing.
        if self.structure == 'fixed':
            # initial block 0:

            # print('Dlatent: ', dlatent.shape)

            # M x 128 x 4 x 4
            x = self.const_input.expand(dlatent.size(0), -1, -1, -1)
            x = x + self.bias.view(1, -1, 1, 1)
            # print('X: ', x.shape)

            # M x 128 x 4 x 4
            x = self.adaIn1(x, self.noise_inputs[0], dlatent[:, 0])
            x = self.conv1(x)
            x = self.adaIn2(x, self.noise_inputs[1], dlatent[:, 1])
            # print('X: ', x.shape)

            # block 1:
            # 128 x 4 x 4 -> 128 x 8 x 8
            x = self.GBlock1(x, dlatent)
            # print('X: ', x.shape)

            # block 2:
            # 128 x 8 x 8 -> 128 x 16 x 16
            x = self.GBlock2(x, dlatent)
            # print('X: ', x.shape)

            # block 3:
            # 128 x 16 x 16 -> 128 x 32 x 32
            x = self.GBlock3(x, dlatent)
            # print('X: ', x.shape)

            # block 4:
            # 128 x 32 x 32 -> 64 x 64 x 64
            x = self.GBlock4(x, dlatent)
            # print('X: ', x.shape)

            # block 5:
            # 64 x 64 x 64 -> 32 x 128 x 128
            x = self.GBlock5(x, dlatent)
            # print('X: ', x.shape)



            # block 6:
            # 32 x 128 x 128 ->  x 128 x 128
            # x = self.GBlock6(x, dlatent)

            # block 7:
            # 64 x 128 x 128 -> 32 x 128 x 128
            # x = self.GBlock7(x, dlatent)

            x = self.channel_shrinkage(x)
            images_out = self.torgb(x)
            return images_out




class StyleGenerator(nn.Module):
    def __init__(self,
                mapping_fmaps=128,
                style_mixing_prob=0.9,       # Probability of mixing styles during training. None = disable.
                truncation_psi=0.7,          # Style strength multiplier for the truncation trick. None = disable.
                truncation_cutoff=8,          # Number of layers for which to apply the truncation trick. None = disable.
                device="cuda",
                **kwargs):
        super(StyleGenerator, self).__init__()
        self.mapping_fmaps = mapping_fmaps
        self.style_mixing_prob = style_mixing_prob
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff

        # print(noise_image.shape if noise_image else "NO Noise Image")
        self.mapping = G_mapping(self.mapping_fmaps, **kwargs)
        self.synthesis = G_synthesis(self.mapping_fmaps, device=device, **kwargs)

    def forward(self, latents1):
        dlatents1, num_layers = self.mapping(latents1)
        dlatents1 = dlatents1.unsqueeze(1)
        dlatents1 = dlatents1.expand(-1, int(num_layers), -1)

        # Apply truncation trick.
        if self.truncation_psi and self.truncation_cutoff:
            coefs = np.ones([1, num_layers, 1], dtype=np.float32)
            for i in range(num_layers):
                if i < self.truncation_cutoff:
                    coefs[:, i, :] *= self.truncation_psi
            """Linear interpolation.
               a + (b - a) * t (a = 0)
               reduce to
               b * t
            """

            dlatents1 = dlatents1 * torch.Tensor(coefs).to(dlatents1.device)
        # print(dlatents1.shape)
        img = self.synthesis(dlatents1)
        return img



# Discriminator

class StyleDiscriminator(nn.Module):
    def __init__(self,
                 resolution=128,
                 fmap_base=128*16,
                 num_channels=3,
                 structure='fixed',  # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, only support 'fixed' mode now.
                 fmap_max=128,
                 fmap_decay=1.0,
                 f=None         # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 ):
        super().__init__()
        self.resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** self.resolution_log2 and resolution >= 4
        # self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        # fromrgb: fixed mode
        self.fromrgb = nn.Conv2d(3, 16, kernel_size=1)
        self.structure = structure

        # blur2d
        self.blur2d = Blur2d(f)

        # down_sample
        self.down1 = nn.AvgPool2d(2)
        self.down21 = nn.Conv2d( 64, 64, kernel_size=2, stride=2)
        # self.down22 = nn.Conv2d(self.nf(self.resolution_log2-6), self.nf(self.resolution_log2-6), kernel_size=2, stride=2)
        # self.down23 = nn.Conv2d(self.nf(self.resolution_log2-7), self.nf(self.resolution_log2-7), kernel_size=2, stride=2)
        # self.down24 = nn.Conv2d(self.nf(self.resolution_log2-8), self.nf(self.resolution_log2-8), kernel_size=2, stride=2)

        # conv1: padding=same
        self.conv1 = nn.Conv2d(16, 16, kernel_size=3, padding=(1, 1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=(1, 1))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=(1, 1))
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=(1, 1))
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=(1, 1))
        # self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=(1, 1))
        # self.conv7 = nn.Conv2d(self.nf(self.resolution_log2-6), self.nf(self.resolution_log2-7), kernel_size=3, padding=(1, 1))
        # self.conv8 = nn.Conv2d(self.nf(self.resolution_log2-7), self.nf(self.resolution_log2-8), kernel_size=3, padding=(1, 1))

        # calculate point:
        self.conv_last = nn.Conv2d(64, 64, kernel_size=3, padding=(1, 1))
        self.dense0 = nn.Linear(1024, 64)
        self.dense1 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        if self.structure == 'fixed':
            # 3 x 128 x 128 -> 16 x 128 x 128
            x = F.leaky_relu(self.fromrgb(input), 0.2, inplace=True)
            
            # 1. 16 x 128 x 128 -> 16 x 64 x 64
            res = self.resolution_log2
            x = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)
            # print(x.shape)

            # 2. 16 x 64 x 64 ->  32 x 32 x 32
            res -= 1
            x = F.leaky_relu(self.conv2(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)
            # print(x.shape)

            # 3. 32 x 32 x 32 -> 64 x 16 x 16
            res -= 1
            x = F.leaky_relu(self.conv3(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)
            # print(x.shape)

            # 4. 64 x 16 x 16 -> 64 x 8 x 8
            res -= 1
            x = F.leaky_relu(self.conv4(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)
            # print(x.shape)

            # 5. 64 x 8 x 8 -> 64 x 4 x 4
            res -= 1
            x = F.leaky_relu(self.conv5(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down21(self.blur2d(x)), 0.2, inplace=True)
            # print(x.shape)

            
            # 9. 64 x 4 x 4 -> 64 x 4 x 4
            x = F.leaky_relu(self.conv_last(x), 0.2, inplace=True)
            

            #  64 x 4 x 4 -> 1024  [N x 8192(4 x 4 x nf(1)).]
            x = x.view(x.size(0), -1)

            # 1024 -> 64
            x = F.leaky_relu(self.dense0(x), 0.2, inplace=True)
            # print(x.shape)
            

            # 64 -> 1 [N x 1]
            x = F.leaky_relu(self.dense1(x), 0.2, inplace=True)
            # print(x.shape)

            return x