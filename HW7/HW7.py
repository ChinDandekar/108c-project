import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import sys
import argparse

import cv2

def openImage(imgname):
    """
    Opens an image file and returns the image object.

    Parameters:
    imgname (str): The path to the image file.

    Returns:
    numpy.ndarray: The image object.

    """
    img = cv2.imread(imgname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def Q1(imgname):
    image = openImage(imgname)
    H = np.array([[1.5, 0.5, 0], [0,2.5,0], [0,0,1]])
    length, width, channels = image.shape
    warpedImage = cv2.warpPerspective(image, H, (length*3, width*2))
    sift = cv2.SIFT_create()
    
    grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    imageKeyPoints, des = sift.detectAndCompute(grayImage,None)
    
    
    grayWarpedImage = cv2.cvtColor(warpedImage, cv2.COLOR_RGB2GRAY)
    
    warpedImageKeyPoints, warpDes = sift.detectAndCompute(grayWarpedImage, None)

    
    image=cv2.drawKeypoints(image,imageKeyPoints,image)
    
    warpedImage = cv2.drawKeypoints(warpedImage, warpedImageKeyPoints, warpedImage)
    
    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(warpedImage)
    plt.title(f"{imgname} warped by matrix:\n {H}")

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Q1', '--Q2', action='store_true', help='If True run Q1 else run Q2')
    parser.add_argument('--imgname', type=str, default='IMG_8833.jpg', help='Name of the image')
    parser.add_argument('--imgname2', type=str, default='IMG_8834.jpg', help='Name of the second image for Q2')
    opt = parser.parse_args()
    
    if opt.Q1:
        Q1(opt.imgname)