import os                               #File management for images
import random as rd                     #Facilitates randomization of training file selection                  
import cv2 as cv                        #Image handling/processing
import numpy as np                      #Support tensorflow (np.array)
import tensorflow as tf                 #Tensorflow AI library
from matplotlib import pyplot as plt    #Plot images and metrics info
from tqdm import tqdm                   #Status bar for long-processing items
import skimage
from utils import get_image, display_comparison


TRAIN_ROOT = 'images/train/'
TEST_ROOT = 'images/test/'
VALIDATE_ROOT = 'images/validate/'
GROUND = 'ground/'
NOISY = 'noisy_gaussian/'

for root in [TRAIN_ROOT,TEST_ROOT,VALIDATE_ROOT]:
    if not (os.path.isdir(root+NOISY)):
        os.makedirs(root+NOISY) 
    files = os.listdir(root+GROUND)
    for file in files:
        img = get_image(root+GROUND+file)
        noise_img = skimage.util.random_noise(img, mode='gaussian',seed=None, clip=True)
        noise_img = np.array(255*noise_img, dtype = 'uint8')
        noise_img = cv.cvtColor(noise_img, cv.COLOR_RGB2BGR)
        cv.imwrite(root+NOISY+file.replace(".jpg", "_noise.jpg" ), noise_img)