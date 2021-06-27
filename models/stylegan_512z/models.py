# stage =  [   0,    1,    2,    3,    4,    5,    6,   7,   8,   9,  10,  11,  12 ]
# nf =     [ 512,  512,  512,  512,  512,  256,  128,  64,  32,  16,   8,   4,   2 ]
#  0  ->  512
#  1  ->  512
#  2  ->  512
#  3  ->  512
#  4  ->  512
#  5  ->  256
#  6  ->  128
#  7  ->   64
#  8  ->   32
#  9  ->   16
# 10  ->    8
# 11  ->    4
# 12  ->    2
# 13  ->    1

from .utils import *

class G_mapping(nn.Module):
    def __init__(self,
                 mapping_fmaps=512,
                 dlatent_size=512,
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
        self.resolution_log2 = int(np.log2(resolution))
        self.num_layers = self.resolution_log2 * 2 - 2
        self.pixel_norm = PixelNorm()

    def forward(self, x):
        if self.normalize_latents:
            x = self.pixel_norm(x)
        out = self.func(x)
        return out, self.num_layers


class G_synthesis(nn.Module):
    def __init__(self,
                 dlatent_size,                       # Disentangled latent (W) dimensionality.
                 resolution=128,                    # Output resolution (1024 x 1024 by default).
                 fmap_base=8192,                     # Overall multiplier for the number of feature maps.
                 num_channels=3,                     # Number of output color channels.
                 structure='fixed',                  # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
                 fmap_max=512,                       # Maximum number of feature maps in any layer.
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

        self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        
        self.structure = structure
        self.resolution_log2 = int(np.log2(resolution))
        # - 2 means we start from feature map with height and width equals 4.
        # as this example, we get num_layers = 18.
        num_layers = self.resolution_log2 * 2 - 2
        self.num_layers = num_layers

        # Noise inputs.
        self.noise_inputs = []
        for layer_idx in range(num_layers):
            res = layer_idx // 2 + 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noise_inputs.append(torch.randn(*shape).to(device))

        # Blur2d
        self.blur = Blur2d(f)

        # torgb: fixed mode
        self.channel_shrinkage = Conv2d(input_channels=self.nf(self.resolution_log2-2),
                                        output_channels=self.nf(self.resolution_log2),
                                        kernel_size=3,
                                        use_wscale=use_wscale)
        self.torgb = Conv2d(self.nf(self.resolution_log2), num_channels, kernel_size=1, gain=1, use_wscale=use_wscale)

        # Initial Input Block
        self.const_input = nn.Parameter(torch.ones(1, self.nf(1), 4, 4))
        self.bias = nn.Parameter(torch.ones(self.nf(1)))
        self.adaIn1 = LayerEpilogue(self.nf(1), dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style)
        self.conv1  = Conv2d(input_channels=self.nf(1), output_channels=self.nf(1), kernel_size=3, use_wscale=use_wscale)
        self.adaIn2 = LayerEpilogue(self.nf(1), dlatent_size, use_wscale, use_noise, use_pixel_norm,
                                    use_instance_norm, use_style)

        # Common Block
        # 4 x 4 -> 8 x 8
        res = 3
        if res <= self.resolution_log2:
            self.GBlock1 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 8 x 8 -> 16 x 16
        res = 4
        if res <= self.resolution_log2:
            self.GBlock2 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 16 x 16 -> 32 x 32
        res = 5
        if res <= self.resolution_log2:
            self.GBlock3 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 32 x 32 -> 64 x 64
        res = 6
        if res <= self.resolution_log2:
            self.GBlock4 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 64 x 64 -> 128 x 128
        res = 7
        if res <= self.resolution_log2:
            self.GBlock5 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 128 x 128 -> 256 x 256
        res = 8
        if res <= self.resolution_log2:
            self.GBlock6 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 256 x 256 -> 512 x 512
        res = 9
        if res <= self.resolution_log2:
            self.GBlock7 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 512 x 512 -> 1024 x 1024
        res = 10
        if res <= self.resolution_log2:
            self.GBlock8 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

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
            # constant input (512 x 4 x 4)
            x = self.const_input.expand(dlatent.size(0), -1, -1, -1)
            x = x + self.bias.view(1, -1, 1, 1)

            # (512 x 4 x 4) -> (512 x 4 x 4)
            x = self.adaIn1(x, self.noise_inputs[0], dlatent[:, 0])
            x = self.conv1(x)
            x = self.adaIn2(x, self.noise_inputs[1], dlatent[:, 1])

            # block 1:
            # (512 x 4 x 4) -> (512 x 8 x 8)
            res = 3
            if res <= self.resolution_log2:
                x = self.GBlock1(x, dlatent)

            # block 2:
            # (512 x 8 x 8) -> (512 x 16 x 16)
            res = 4
            if res <= self.resolution_log2:
                x = self.GBlock2(x, dlatent)

            # block 3:
            # (512 x 16 x 16) -> (512 x 32 x 32)
            res = 5
            if res <= self.resolution_log2:
                x = self.GBlock3(x, dlatent)

            # block 4:
            # (512 x 32 x 32) -> (256 x 64 x 64)
            res = 6
            if res <= self.resolution_log2:
                x = self.GBlock4(x, dlatent)

            # block 5:
            # (256 x 64 x 64) -> (128 x 128 x 128)
            res = 7
            if res <= self.resolution_log2:
                x = self.GBlock5(x, dlatent)

            # block 6:
            # (128 x 128 x 128) -> (64 x 256 x 256)
            res = 8
            if res <= self.resolution_log2:
                x = self.GBlock6(x, dlatent)

            # block 7:
            # (64 x 256 x 256) -> (32 x 512 x 512)
            res = 9
            if res <= self.resolution_log2:
                x = self.GBlock7(x, dlatent)

            # block 8:
            # (32 x 512 x 512) -> (16 x 1024 x 1024)
            res = 10
            if res <= self.resolution_log2:
                x = self.GBlock8(x, dlatent)

            x = self.channel_shrinkage(x)
            # (16 x 1024 x 1024) -> (3 x 1024 x 1024)
            images_out = self.torgb(x)
            return images_out


class StyleGenerator(nn.Module):
    def __init__(self,
                 mapping_fmaps=512,
                 style_mixing_prob=0.9,       # Probability of mixing styles during training. None = disable.
                 truncation_psi=0.7,          # Style strength multiplier for the truncation trick. None = disable.
                 truncation_cutoff=8,          # Number of layers for which to apply the truncation trick. None = disable.
                 device="cuda",
                 **kwargs
                 ):
        super(StyleGenerator, self).__init__()
        self.mapping_fmaps = mapping_fmaps
        self.style_mixing_prob = style_mixing_prob
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff

        self.mapping = G_mapping(self.mapping_fmaps, **kwargs)
        self.synthesis = G_synthesis(self.mapping_fmaps, device=device, **kwargs)

    def forward(self, latents1):
        dlatents1, num_layers = self.mapping(latents1)
        # let [N, O] -> [N, num_layers, O]
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

        img = self.synthesis(dlatents1)
        return img


class StyleDiscriminator(nn.Module):
    def __init__(self,
                 resolution=1024,
                 fmap_base=8192,
                 num_channels=3,
                 structure='fixed',  # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, only support 'fixed' mode now.
                 fmap_max=512,
                 fmap_decay=1.0,
                 # f=[1, 2, 1]         # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 f=None         # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 ):
        """
            Noitce: we only support input pic with height == width.

            if H or W >= 128, we use avgpooling2d to do feature map shrinkage.
            else: we use ordinary conv2d.
        """
        super().__init__()
        self.resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** self.resolution_log2 and resolution >= 4
        self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        
        self.structure = structure
        
        # fromrgb: fixed mode 
        self.fromrgb = nn.Conv2d(
                            num_channels, 
                            self.nf(self.resolution_log2-1), 
                            kernel_size=1)

        # blur2d
        self.blur2d = Blur2d(f)

        # down_sample
        self.down1 = nn.AvgPool2d(2)
        self.down21 = nn.Conv2d(self.nf(self.resolution_log2-5), self.nf(self.resolution_log2-5), kernel_size=2, stride=2)
        self.down22 = nn.Conv2d(self.nf(self.resolution_log2-6), self.nf(self.resolution_log2-6), kernel_size=2, stride=2)
        self.down23 = nn.Conv2d(self.nf(self.resolution_log2-7), self.nf(self.resolution_log2-7), kernel_size=2, stride=2)
        self.down24 = nn.Conv2d(self.nf(self.resolution_log2-8), self.nf(self.resolution_log2-8), kernel_size=2, stride=2)

        # conv1: padding=same
        if self.resolution_log2 > 2:
            self.conv1 = nn.Conv2d(self.nf(self.resolution_log2-1), self.nf(self.resolution_log2-1), kernel_size=3, padding=(1, 1))
        if self.resolution_log2 > 3:
            self.conv2 = nn.Conv2d(self.nf(self.resolution_log2-1), self.nf(self.resolution_log2-2), kernel_size=3, padding=(1, 1))
        
        if self.resolution_log2 > 4:
            self.conv3 = nn.Conv2d(self.nf(self.resolution_log2-2), self.nf(self.resolution_log2-3), kernel_size=3, padding=(1, 1))
        
        if self.resolution_log2 > 5:
            self.conv4 = nn.Conv2d(self.nf(self.resolution_log2-3), self.nf(self.resolution_log2-4), kernel_size=3, padding=(1, 1))
        
        if self.resolution_log2 > 6:
            self.conv5 = nn.Conv2d(self.nf(self.resolution_log2-4), self.nf(self.resolution_log2-5), kernel_size=3, padding=(1, 1))
        
        if self.resolution_log2 > 7:
            self.conv6 = nn.Conv2d(self.nf(self.resolution_log2-5), self.nf(self.resolution_log2-6), kernel_size=3, padding=(1, 1))
        
        if self.resolution_log2 > 8:
            self.conv7 = nn.Conv2d(self.nf(self.resolution_log2-6), self.nf(self.resolution_log2-7), kernel_size=3, padding=(1, 1))
        
        if self.resolution_log2 > 9:
            self.conv8 = nn.Conv2d(self.nf(self.resolution_log2-7), self.nf(self.resolution_log2-8), kernel_size=3, padding=(1, 1))

        # calculate point:
        self.conv_last = nn.Conv2d(self.nf(self.resolution_log2-8), self.nf(1), kernel_size=3, padding=(1, 1))
        self.dense0 = nn.Linear(fmap_base, self.nf(0))
        self.dense1 = nn.Linear(self.nf(0), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        if self.structure == 'fixed':
            
            # ( 3 x res x res ) -> ( 512 x res x res )
            x = F.leaky_relu(self.fromrgb(input), 0.2, inplace=True)
            
            # 1. 1024 x 1024 x nf(9)(16) -> 512 x 512
            res = self.resolution_log2

            if res >= 3:
                x = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
                x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)

            # 2. 512 x 512 -> 256 x 256
            res -= 1
            
            if res >= 3:
                x = F.leaky_relu(self.conv2(x), 0.2, inplace=True)
                x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)

            # 3. 256 x 256 -> 128 x 128
            res -= 1
            if res >= 3:
                x = F.leaky_relu(self.conv3(x), 0.2, inplace=True)
                x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)

            # 4. 128 x 128 -> 64 x 64
            res -= 1

            if res >= 3:
                x = F.leaky_relu(self.conv4(x), 0.2, inplace=True)
                x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)

            # 5. 64 x 64 -> 32 x 32
            res -= 1

            if res >= 3:
                x = F.leaky_relu(self.conv5(x), 0.2, inplace=True)
                x = F.leaky_relu(self.down21(self.blur2d(x)), 0.2, inplace=True)

            # 6. 32 x 32 -> 16 x 16
            res -= 1
            if res >= 3:
                x = F.leaky_relu(self.conv6(x), 0.2, inplace=True)
                x = F.leaky_relu(self.down22(self.blur2d(x)), 0.2, inplace=True)

            # 7. 16 x 16 -> 8 x 8
            res -= 1
            if res >= 3:
                x = F.leaky_relu(self.conv7(x), 0.2, inplace=True)
                x = F.leaky_relu(self.down23(self.blur2d(x)), 0.2, inplace=True)

            # 8. 8 x 8 -> 4 x 4
            res -= 1
            if res >= 3:
                x = F.leaky_relu(self.conv8(x), 0.2, inplace=True)
                x = F.leaky_relu(self.down24(self.blur2d(x)), 0.2, inplace=True)

            # 9. 4 x 4 -> point
            x = F.leaky_relu(self.conv_last(x), 0.2, inplace=True)
            # N x 8192(4 x 4 x nf(1)).
            x = x.view(x.size(0), -1)
            x = F.leaky_relu(self.dense0(x), 0.2, inplace=True)
            # N x 1
            x = F.leaky_relu(self.dense1(x), 0.2, inplace=True)
            return x

