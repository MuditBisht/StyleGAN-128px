import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from .utils import PixelNormLayer, MyLinear, GSynthesisBlock, MyConv2d, InputBlock, BlurLayer
from collections import OrderedDict

class G_mapping(nn.Sequential):
    def __init__(self, nonlinearity='lrelu', use_wscale=True):
        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]
        layers = [
            ('pixel_norm', PixelNormLayer()),
            ('dense0', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense0_act', act),
            ('dense1', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense1_act', act),
            ('dense2', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense2_act', act),
            ('dense3', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense3_act', act),
            ('dense4', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense4_act', act),
            ('dense5', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense5_act', act),
            ('dense6', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense6_act', act),
            ('dense7', MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale)),
            ('dense7_act', act)
        ]
        super().__init__(OrderedDict(layers))
        
    def forward(self, x):
        x = super().forward(x)
        # Broadcast
        x = x.unsqueeze(1).expand(-1, 18, -1)
        return x


class G_synthesis(nn.Module):
    def __init__(self,
        dlatent_size        = 512,          # Disentangled latent (W) dimensionality.
        num_channels        = 3,            # Number of output color channels.
        resolution          = 1024,         # Output resolution.
        fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
        fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
        fmap_max            = 512,          # Maximum number of feature maps in any layer.
        use_styles          = True,         # Enable style inputs?
        const_input_layer   = True,         # First layer is a learned constant?
        use_noise           = True,         # Enable noise inputs?
        randomize_noise     = True,         # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
        nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu'
        use_wscale          = True,         # Enable equalized learning rate?
        use_pixel_norm      = False,        # Enable pixelwise feature vector normalization?
        use_instance_norm   = True,         # Enable instance normalization?
        dtype               = torch.float32,  # Data type to use for activations and outputs.
        blur_filter         = [1,2,1],      # Low-pass filter to apply when resampling activations. None = no filtering.
        ):
        
        super().__init__()
        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        self.dlatent_size = dlatent_size
        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2**resolution_log2 and resolution >= 4

        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]
        num_layers = resolution_log2 * 2 - 2
        num_styles = num_layers if use_styles else 1
        torgbs = []
        blocks = []
        for res in range(2, resolution_log2 + 1):
            channels = nf(res-1)
            name = '{s}x{s}'.format(s=2**res)
            if res == 2:
                blocks.append((name,
                               InputBlock(channels, dlatent_size, const_input_layer, gain, use_wscale,
                                      use_noise, use_pixel_norm, use_instance_norm, use_styles, act)))
                
            else:
                blocks.append((name,
                               GSynthesisBlock(last_channels, channels, blur_filter, dlatent_size, gain, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, act)))
            last_channels = channels
        self.torgb = MyConv2d(channels, num_channels, 1, gain=1, use_wscale=use_wscale)
        self.blocks = nn.ModuleDict(OrderedDict(blocks))
        
    def forward(self, dlatents_in):
        # Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size].
        # lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0), trainable=False), dtype)
        batch_size = dlatents_in.size(0)       
        for i, m in enumerate(self.blocks.values()):
            if i == 0:
                x = m(dlatents_in[:, 2*i:2*i+2])
            else:
                x = m(x, dlatents_in[:, 2*i:2*i+2])
        rgb = self.torgb(x)
        return rgb



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
        # self.blur2d = BlurLayer(f)
        self.blur2d = lambda x: x

        # conv1: padding=same
        # (16 x 1024 x 1024 ) -> ( 32 x 512 x 512 )
        if self.resolution_log2 >= 10:
            self.conv10 = nn.Conv2d( 16, 16, kernel_size=3, padding=(1, 1))
            self.conv11 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=(1,1))
        # ( 32 x 512 x 512 ) -> ( 64 x 256 x 256 )
        if self.resolution_log2 >= 9:
            self.conv20 = nn.Conv2d(32, 32, kernel_size=3, padding=(1, 1))
            self.conv21 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=(1,1))
        
        # ( 64 x 256 x 256 ) -> ( 128 x 128 x 128 )
        if self.resolution_log2 >= 8:
            self.conv30 = nn.Conv2d(64, 64, kernel_size=3, padding=(1, 1))
            self.conv31 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=(1, 1))
            
        # ( 128 x 128 x 128 ) -> ( 256 x 64 x 64 )
        if self.resolution_log2 >= 7:
            self.conv40 = nn.Conv2d( 128, 128, kernel_size=3, padding=(1, 1))
            self.conv41 = nn.Conv2d( 128, 256, kernel_size=3, stride=2, padding=(1, 1))
            
        
        # ( 256 x 64 x 64 ) -> ( 512 x 32 x 32 )
        if self.resolution_log2 >= 6:
            self.conv50 = nn.Conv2d(256, 256, kernel_size=3, padding=(1, 1))
            self.conv51 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=(1, 1))
        
        # ( 512 x 32 x 32 ) -> ( 512 x 16 x 16 )
        if self.resolution_log2 >= 5:
            self.conv60 = nn.Conv2d(512, 512, kernel_size=3, padding=(1, 1))
            self.conv61 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=(1, 1))
            
        
        # ( 512 x 16 x 16 ) -> ( 512 x 8 x 8 )
        if self.resolution_log2 >= 4:
            self.conv70 = nn.Conv2d(512, 512, kernel_size=3, padding=(1, 1))
            self.conv71 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=(1, 1))

        
        # ( 512 x 8 x 8 ) -> ( 512 x 4 x 4 )
        if self.resolution_log2 >= 3:
            self.conv80 = nn.Conv2d( 512, 512, kernel_size=3, padding=(1, 1))
            self.conv81 = nn.Conv2d( 512, 512, kernel_size=3, stride=2, padding=(1, 1))
            

        # calculate point:
        self.conv_last = nn.Conv2d( 512, 512, kernel_size=3, padding=(1, 1))
        
        
        self.dense0 = nn.Linear(fmap_base, self.nf(0))
        self.dense1 = nn.Linear(self.nf(0), 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        if self.structure == 'fixed':
            
            # ( 3 x res x res ) -> ( 512 x res x res )
            x = F.leaky_relu(self.fromrgb(input), 0.2, inplace=True)
            
            # 1. 1024 x 1024 x nf(9)(16) -> 512 x 512
            # res = 10
            if self.resolution_log2 >= 10:
                x = F.leaky_relu(self.conv10(x), 0.2, inplace=True)
                x = F.leaky_relu(self.conv11(self.blur2d(x)), 0.2, inplace=True)

            # 2. 512 x 512 -> 256 x 256
            # res = 9
            if self.resolution_log2 >= 9:
                x = F.leaky_relu(self.conv20(x), 0.2, inplace=True)
                x = F.leaky_relu(self.conv21(self.blur2d(x)), 0.2, inplace=True)

            # 3. 256 x 256 -> 128 x 128
            # res = 8
            if self.resolution_log2 >= 8:
                x = F.leaky_relu(self.conv30(x), 0.2, inplace=True)
                x = F.leaky_relu(self.conv31(self.blur2d(x)), 0.2, inplace=True)

            # 4. 128 x 128 -> 64 x 64
            # res = 7
            if self.resolution_log2 >= 7:
                x = F.leaky_relu(self.conv40(x), 0.2, inplace=True)
                x = F.leaky_relu(self.conv41(self.blur2d(x)), 0.2, inplace=True)

            # 5. 64 x 64 -> 32 x 32
            # res = 6
            if self.resolution_log2 >= 6:
                x = F.leaky_relu(self.conv50(x), 0.2, inplace=True)
                x = F.leaky_relu(self.conv51(self.blur2d(x)), 0.2, inplace=True)

            # 6. 32 x 32 -> 16 x 16
            # res = 5
            if self.resolution_log2 >= 5:
                x = F.leaky_relu(self.conv60(x), 0.2, inplace=True)
                x = F.leaky_relu(self.conv61(self.blur2d(x)), 0.2, inplace=True)

            # 7. 16 x 16 -> 8 x 8
            # res = 4
            if self.resolution_log2 >= 4:
                x = F.leaky_relu(self.conv70(x), 0.2, inplace=True)
                x = F.leaky_relu(self.conv71(self.blur2d(x)), 0.2, inplace=True)

            # 8. 8 x 8 -> 4 x 4
            # res = 3
            if self.resolution_log2 >= 3:
                x = F.leaky_relu(self.conv80(x), 0.2, inplace=True)
                x = F.leaky_relu(self.conv81(self.blur2d(x)), 0.2, inplace=True)

            # 9. 4 x 4 -> point
            x = F.leaky_relu(self.conv_last(x), 0.2, inplace=True)

            # N x 8192(4 x 4 x nf(1)).
            x = x.view(x.size(0), -1)
            x = F.leaky_relu(self.dense0(x), 0.2, inplace=True)
            # N x 1
            x = F.leaky_relu(self.dense1(x), 0.2, inplace=True)
            return x



def getGenerator(res):
    Generator = nn.Sequential(OrderedDict([
        ('g_mapping', G_mapping()),
        #('truncation', Truncation(avg_latent)),
        ('g_synthesis', G_synthesis(resolution=res))
    ]))
    return Generator

def key_translate(k):
    k = k.lower().split('/')
    if k[0] == 'g_synthesis':
        if not k[1].startswith('torgb'):
            k.insert(1, 'blocks')
        k = '.'.join(k)
        k = (k.replace('const.const','const').replace('const.bias','bias').replace('const.stylemod','epi1.style_mod.lin')
              .replace('const.noise.weight','epi1.top_epi.noise.weight')
              .replace('conv.noise.weight','epi2.top_epi.noise.weight')
              .replace('conv.stylemod','epi2.style_mod.lin')
              .replace('conv0_up.noise.weight', 'epi1.top_epi.noise.weight')
              .replace('conv0_up.stylemod','epi1.style_mod.lin')
              .replace('conv1.noise.weight', 'epi2.top_epi.noise.weight')
              .replace('conv1.stylemod','epi2.style_mod.lin')
              .replace('torgb_lod0','torgb'))
    else:
        k = '.'.join(k)
    return k

def weight_translate(k, w):
    k = key_translate(k)
    if k.endswith('.weight'):
        if w.dim() == 2:
            w = w.t()
        elif w.dim() == 1:
            pass
        else:
            assert w.dim() == 4
            w = w.permute(3, 2, 0, 1)
    return w



def getPretrainedGenerator(resolution, path):
    # path = os.path.join(
    #     os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    #     "weights", "karras2019stylegan-ffhq-1024x1024.pt")
    _, _, sGs = torch.load(path)

    r = int(np.log2(resolution + 1))
    
    param_dict = {key_translate(k): weight_translate(k, v) for k, v in sGs.items() if 'torgb_lod' not in key_translate(k)}
    
    k = f'G_synthesis/ToRGB_lod{10-r}/weight'
    v = sGs[k]
    w = weight_translate(k, v)
    k = f'G_synthesis/ToRGB_lod{10-r}/bias'
    v = sGs[k]
    b = weight_translate(k, v)

    param_dict['g_synthesis.torgb.weight'] = w
    param_dict['g_synthesis.torgb.bias'] = b
    G = getGenerator(resolution)
    G.load_state_dict(param_dict, strict=False)
    return G

def getPretrainedDiscriminator(res, path):
    weights_pt = torch.load(path)
    _, sD, _= weights_pt

    r = int(np.log2(res + 1))
    D = StyleDiscriminator(resolution=res)
    d = D.state_dict()

    d['fromrgb.weight'] = sD[f"FromRGB_lod{10-r}/weight"].permute(3,2,0,1)
    d['fromrgb.bias'] = sD[f"FromRGB_lod{10-r}/bias"]


    d['conv_last.weight'] = sD["4x4/Conv/weight"].permute(3,2,0,1)
    d['conv_last.weight'] = d['conv_last.weight'][:, :512, : , : ]

    d['conv_last.bias'] = sD["4x4/Conv/bias"]
    for i in range(1, r-1):
        if ( d[f'conv{9-i}0.weight'].shape == sD[f"{2**(i+2)}x{2**(i+2)}/Conv0/weight"].permute(3,2,0,1).shape and
             d[f'conv{9-i}0.bias'].shape ==  sD[f"{2**(i+2)}x{2**(i+2)}/Conv0/bias"].shape and
             d[f'conv{9-i}1.weight'].shape == sD[f"{2**(i+2)}x{2**(i+2)}/Conv1_down/weight"].permute(3,2,0,1).shape and
             d[f'conv{9-i}1.bias'].shape ==  sD[f"{2**(i+2)}x{2**(i+2)}/Conv1_down/bias"].shape ):
            d[f'conv{9-i}0.weight'] = sD[f"{2**(i+2)}x{2**(i+2)}/Conv0/weight"].permute(3,2,0,1)
            d[f'conv{9-i}0.bias'] =  sD[f"{2**(i+2)}x{2**(i+2)}/Conv0/bias"]
            d[f'conv{9-i}1.weight'] = sD[f"{2**(i+2)}x{2**(i+2)}/Conv1_down/weight"].permute(3,2,0,1)
            d[f'conv{9-i}1.bias'] =  sD[f"{2**(i+2)}x{2**(i+2)}/Conv1_down/bias"]
            print(f'conv{9-i}')
        else:
            print("Error on", i)
            raise Exception("error")

    d["dense0.weight"] = sD["4x4/Dense0/weight"].permute(1,0)
    d["dense0.bias"] = sD["4x4/Dense0/bias"]
    d["dense1.weight"] = sD["4x4/Dense1/weight"].permute(1,0)
    d["dense1.bias"] = sD["4x4/Dense1/bias"]

    D.load_state_dict(d)
    return D