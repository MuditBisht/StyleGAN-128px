import torch
import numpy as np
import cv2 as cv

from utils.img_transformation  import *
from model import models

G = models.StyleGenerator(device="cpu")

print("Loading weights ...")
load_old_path = f'weights/model-330-cpu.pth'
state = torch.load(load_old_path)
G.load_state_dict(state['G'])

z = torch.rand(1, 128)
print("Generating image.....")
img = G.forward(z)
img = UnNormalizeImage(img)
img = img[0].T.detach().to(dtype=torch.uint8).numpy()
img = cv.resize(img[:,:,::-1], (256, 256))


key = 0
while key != ord('q'):
    key = cv.waitKey(10)
    if key == ord('n'):
        z = torch.rand(1, 128)
        print("Generating image.....")
        img = G.forward(z)
        img = UnNormalizeImage(img)
        img = img[0].T.detach().to(dtype=torch.uint8).numpy()
        img = cv.resize(img[:,:,::-1], (256, 256))
        
    cv.imshow("image", img)


