import os                               #File management for images
import random as rd                     #Facilitates randomization of training file selection                  
import cv2 as cv                        #Image handling/processing
import numpy as np                      #Support tensorflow (np.array)
import tensorflow as tf                 #Tensorflow AI library
from matplotlib import pyplot as plt    #Plot images and metrics info
from tqdm import tqdm                   #Status bar for long-processing items
import skimage

#Non-Local Means Denoising ------------------------------------------------------------------------
def NLM(image):
    #Wrapper function on openCV NLM
    #TODO - tune parameters/config
    return cv.fastNlMeansDenoisingColored(image,None,35,35,7,21)


#Image Pre/Post-Processing ------------------------------------------------------------------------
def get_image(filename):
    #Wrapper function on cv.imread. Automatically converts from BGR to RGB
    image = cv.imread(filename)
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)

def normalize(pixel):
    #Scale input pixel from int [0,255] to float32 [0,1] to normalize before sending to NN
    #DO NOT USE - use vectorized v_normalize(). Operates over all elements in np.array
    return np.float32(pixel/255.0)
v_normalize = np.vectorize(normalize)   #Vectorize for use over np.array

def denormalize(pixel):
    #De-scale input pixel from float32 [0,1] to int [0,255] to denormalize image from NN
    #DO NOT USE - use vectorized v_denormalize(). Operates over all elements in np.array
    return int(pixel*255)
v_denormalize = np.vectorize(denormalize)   #Vectorize for use over np.array


def gen_patches(image,n=8):
    #Split input image into np.array of sub-image patches
    #Splits into n x n sub-images. Default n=8 yeilding 64 sub-images

    #Calculate patch dimensions and initialize empty np.array
    w,h,d = image.shape
    w = w//n
    h = h//n
    patches = np.zeros((n*n,w,h,d))

    #Iterate over sub-images and store to np.array
    for i in range(n):
        for j in range(n):
            x = i*w
            y = j*h
            patches[i*n+j] = image[x:x+w, y:y+h, :]
    return patches

def stitch_patches(patches,n=8):
    #Recombine sub-images into complete image. Takes np.array with sub-images as 0th dimension
    #Recombines image from n x n sub-images. Default n=8 expects 64 sub-images

    #Calculate image dimensions and initialize image np.array
    w,h,d = patches[0].shape
    W = w*n
    H = h*n
    image = np.zeros((W,H,d), dtype='float32')

    #Iterate over sub-images and recombine to final image
    for i in range(len(patches)):
        x = i//n
        y = i%n
        image[x*w:(x+1)*w, y*h:(y+1)*h, :] = patches[i]
    return image

#Performance Metrics ------------------------------------------------------------------------------
#TODO - create performance metrics functions to be used on denoising analysis
def MSE(A,B):
    #TODO - mse implementation. Can reuse tf.keras.losses.MeanSquaredError() ?
    return 0

def PNSR(A,B):
    #TODO - psnr (peak signal to noise ratio) implementation
    # tf.image.psnr(a, b, max_val, name=None)
    return 0

def SSIM(A,B):
    #TODO - ssim (structural similarity index) implementation
    #tf.image.ssim(
    return 0

#Display Functions --------------------------------------------------------------------------------
def display_image(image):
    #Simple image display via matplotlib
    plt.close()
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    return

def display_comparison(images):
    #Simple display bis matplot lib, compare images (visually) by plotting
    #them side by side
    plt.close()
    fig,ax = plt.subplots(1,len(images))
    for i in range(len(images)):
        ax[i].imshow(images[i])
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
    plt.show()
    return


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def mse(img1, img2):
    h, w = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff**2)
    mse = err/(float(h*w))
    return mse