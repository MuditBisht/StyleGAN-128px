import pickle, torch, collections
import cv2 as cv
import numpy as np
from models import G_mapping, G_synthesis, getPretrainedGenerator, getGenerator

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




def main():
    weights_pt = torch.load('../../weights/karras2019stylegan-ffhq-1024x1024.pt')
    sG, sD, sGs = weights_pt
    param_dict = {key_translate(k): weight_translate(k, v) for k, v in sGs.items() if 'torgb_lod' not in key_translate(k)}
    
    k = 'G_synthesis/ToRGB_lod0/weight'
    v = sGs[k]
    w = weight_translate(k, v)
    k = 'G_synthesis/ToRGB_lod0/bias'
    v = sGs[k]
    b = weight_translate(k, v)

    param_dict['g_synthesis.torgb.weight'] = w
    param_dict['g_synthesis.torgb.bias'] = b
    G = getGenerator(1024)
    G.load_state_dict(param_dict, strict=False)


    latents = torch.randn(1, 512)
    imgs = G(latents)
    imgs = (imgs.clamp(-1, 1) + 1) / 2.0 # normalization to 0..1 range
    img = imgs[0].detach().numpy()
    img = cv.cvtColor(np.einsum('ijk->jki', img), cv.COLOR_RGB2BGR)
    key = 0
    while key != ord('q'):
        key = cv.waitKey(10)
        if key == ord('n'):
            z = torch.rand(1, 512)
            print("Generating image.....")
            img = G.forward(z)[0]
            img = (img.clamp(-1, 1) + 1) / 2.0
            img = img.detach().numpy()
            img = cv.cvtColor(np.einsum('ijk->jki', img), cv.COLOR_RGB2BGR)
        cv.imshow("image", img)

def mainv2():
    res = 128
    G = getPretrainedGenerator(res)
    latents = torch.randn(1, 512)
    imgs = G(latents)
    imgs = (imgs.clamp(-1, 1) + 1) / 2.0 # normalization to 0..1 range
    img = imgs[0].detach().numpy()
    img = cv.cvtColor(np.einsum('ijk->jki', img), cv.COLOR_RGB2BGR)
    key = 0
    while key != ord('q'):
        key = cv.waitKey(10)
        if key == ord('n'):
            z = torch.rand(1, 512)
            print("Generating image.....")
            img = G.forward(z)[0]
            img = (img.clamp(-1, 1) + 1) / 2.0
            img = img.detach().numpy()
            img = cv.cvtColor(np.einsum('ijk->jki', img), cv.COLOR_RGB2BGR)
        cv.imshow("image", img)

if __name__ == '__main__':
    mainv2()