import sys
import argparse
import torch
from models.stylegan_keras import getGenerator, getPretrainedGenerator, G_mapping, G_synthesis
import cv2
import numpy as np
import os

def save_images(images, dir):
    c = -5
    for img in images:
        img = np.einsum('ijk->jki', img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = (img * 255).astype(np.uint8)
        fl = os.path.join(dir, f"image({c})")
        print('Saving ', fl)
        cv2.imwrite(fl, img)
        c+=1

def save_image(image, fl):
    img = np.einsum('ijk->jki', image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = (img * 255).astype(np.uint8)
    print('Saving ', fl)
    cv2.imwrite(fl, img)


def generateBoundaryZ(bound, out):
    G = None
    z = torch.rand(1, 512)*3
    
    wt_path = './weights/stylegan-1024-generator.pt'
    g = torch.load(wt_path)
    G = getGenerator(1024)
    G.load_state_dict(g)
    d = None
    if bound == 'age':
        d = torch.FloatTensor(np.load("./weights/boundaries/stylegan_ffhq_age_boundary.npy"))
    elif bound == 'gender':
        d = torch.FloatTensor(np.load("./weights/boundaries/stylegan_ffhq_gender_boundary.npy"))
    elif bound == 'eyeglasses':
        d = torch.FloatTensor(np.load("./weights/boundaries/stylegan_ffhq_eyeglasses_boundary.npy"))
    elif bound == 'pose':
        d = torch.FloatTensor(np.load("./weights/boundaries/stylegan_ffhq_pose_boundary.npy"))
    elif bound == 'smile':
        d = torch.FloatTensor(np.load("./weights/boundaries/stylegan_ffhq_smile_boundary.npy"))
    else:
        print('Invalid boundary')
        return
    
    Z = []
    for i in range(-5, 6):
        z_ = z + d*1*i
        img = G.forward(z_)[0]
        img = ((img.clamp(-1, 1) + 1) / 2.0).detach() # normalization to 0..1 range
        fl = os.path.join(out, f"image({i}).jpeg")
        save_image(img, fl)


def generateBoundaryW(bound, out):
    G = None
    z = torch.rand(1, 512)
    
    wt_path = './weights/stylegan-1024-generator.pt'
    g = torch.load(wt_path)
    G = getGenerator(1024)
    G.load_state_dict(g)
    d = None
    if bound == 'age':
        d = torch.FloatTensor(np.load("./weights/boundaries/stylegan_ffhq_age_boundary.npy"))
    elif bound == 'gender':
        d = torch.FloatTensor(np.load("./weights/boundaries/stylegan_ffhq_gender_boundary.npy"))
    elif bound == 'eyeglasses':
        d = torch.FloatTensor(np.load("./weights/boundaries/stylegan_ffhq_eyeglasses_boundary.npy"))
    elif bound == 'pose':
        d = torch.FloatTensor(np.load("./weights/boundaries/stylegan_ffhq_pose_boundary.npy"))
    elif bound == 'smile':
        d = torch.FloatTensor(np.load("./weights/boundaries/stylegan_ffhq_smile_boundary.npy"))
    else:
        print('Invalid boundary')
        return
    
    Z = []
    for i in range(-5, 6):
        z_ = z + d*1*i
        img = G.forward(z_)[0]
        img = ((img.clamp(-1, 1) + 1) / 2.0).detach() # normalization to 0..1 range
        fl = os.path.join(out, f"image({i}).jpeg")
        save_image(img, fl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--latent-space', choices=["z", "w"])
    parser.add_argument('-b', '--boundary', choices=["age", "eyeglasses", "gender", "pose", "smile"])
    parser.add_argument('-o', '--out')
    args = parser.parse_args()
    
    bound = args.boundary
    out = args.out if args.out else "images"

    print('res: ', bound)
    print('out: ', out)
    
    generateBoundaryZ(bound, out)
    generateBoundaryW(bound, out)