import sys
import argparse
import torch
from models.stylegan_keras import getGenerator, getPretrainedGenerator
import cv2
import numpy as np

def save_image(images, fl):
    new_imgs = []
    for img in images:
        img = np.einsum('ijk->jki', img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        new_imgs.append(torch.tensor(img))
    images = torch.stack(new_imgs).numpy()
    print('Image: ', images.shape)
    img1 = cv2.vconcat(images[:3, :])
    img2 = cv2.vconcat(images[3:6, :])
    img3 = cv2.vconcat(images[6:, :])
    img = cv2.hconcat([img1, img2, img3])
    print('Image: ', img.shape)

    img = (img * 255).astype(np.uint8)
    cv2.imwrite(fl, img)
    return img

def generateImage(res, out):
    G = None
    z = torch.rand(9, 512)

    if res == '128':
        wt_path = './weights/stylegan-128-generator.pt'
        g = torch.load(wt_path)
        G = getGenerator(128)
        G.load_state_dict(g)
    elif res == '1024':
        wt_path = './weights/stylegan-1024-generator.pt'
        g = torch.load(wt_path)
        G = getGenerator(1024)
        G.load_state_dict(g)
    else:
        print('Invalid pixels')
        return
    
    imgs = G.forward(z)
    imgs = ((imgs.clamp(-1, 1) + 1) / 2.0).detach() # normalization to 0..1 range
    
    img = save_image(imgs, out)
    key = 0
    while key != ord('q'):
        key = cv2.waitKey(10)
        cv2.imshow("image", img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resolution', choices=["128", "256", "512", "1024"])
    parser.add_argument('-o', '--out')
    args = parser.parse_args()
    
    res = args.resolution
    out = args.out if args.out else "./image.jpeg"

    print('res: ', res)
    print('out: ', out)
    
    generateImage(res, out)