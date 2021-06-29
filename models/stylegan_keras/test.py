import models
import torch
import cv2 as cv
import numpy as np

def main():
    G = models.getPretrainedGAN(128)
    z = torch.rand(1, 512)
    print("Generating image.....")
    img = G.forward(z)[0]
    img = (img.clamp(-1, 1) + 1) / 2.0
    img = img.detach().numpy()
    print(img.shape)
    img = cv.cvtColor(np.einsum('ijk->jki', img), cv.COLOR_RGB2BGR)
    print(img.shape)
    key = 0
    while key != ord('q'):
        key = cv.waitKey(10)
        if key == ord('n'):
            z = torch.rand(1, 512)
            print("Generating image.....")
            img = G.forward(z)[0]
            img = (img.clamp(-1, 1) + 1) / 2.0
            img = img.detach().numpy()
            print(img.shape)
            img = cv.cvtColor(np.einsum('ijk->jki', img), cv.COLOR_RGB2BGR)
            print(img.shape)
        cv.imshow("image", img)

main()

