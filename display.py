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

files = os.listdir(TEST_ROOT+NOISY)
files = rd.choices(files,k=6)
for file in files:
    noisy = get_image(TEST_ROOT+NOISY+file)
    ground = get_image(TEST_ROOT+GROUND+file.replace("_noise.jpg", ".jpg"))
    display_comparison([ground,noisy])