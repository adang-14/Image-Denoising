'''
=========================================================================================================
Description:    Denoising Autoencoder Analysis
Purpose:        Analysis of image denoising via autoencoder neural-net machine learning. Below content
                contains autoencoder implementation, competing methods (NLM), and performance
Python:         Version 3.9
Authors:        Andy Dang, Tayte Waterman
Date:           ---
=========================================================================================================
'''
#Imports ================================================================================================
import os                               #File management for images
import random as rd                     #Facilitates randomization of training file selection                  
import cv2 as cv                        #Image handling/processing
import numpy as np                      #Support tensorflow (np.array)
import tensorflow as tf                 #Tensorflow AI library
from matplotlib import pyplot as plt    #Plot images and metrics info
from tqdm import tqdm                   #Status bar for long-processing items
import skimage
from utils import *

#Constants ==============================================================================================
#Image folder structure ---------------------------------------------------------------------------
TRAIN_ROOT = 'images/train/'
TEST_ROOT = 'images/test/'
VALIDATE_ROOT = 'images/validate/'
GROUND = 'ground/'
NOISY = 'noisy_gaussian/'

#Non-Local Means Denoising ------------------------------------------------------------------------
def NLM(image):
    #Wrapper function on openCV NLM
    return cv.fastNlMeansDenoisingColored(image,None,35,35,7,21)

#CNN Autoencoder ----------------------------------------------------------------------------------
class Autoencoder:
    #       initial set-up as proof-of-concept/image testing. Has not been optimized
    def __init__(self):
        #Constructor - Initialize NN structure and compile TensorFlow model. Attempts
        #              to reload previously trained model weights if available and
        #              compatible

        #NN Model structure
        input = tf.keras.Input(shape=(64,64,3)) #64 x 64 patch images
        #Encoder
        x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(input)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
        #Decoder
        x = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu",kernel_initializer='he_normal', padding="same")(x)
        x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", kernel_initializer='he_normal',padding="same")(x)
        x = tf.keras.layers.Conv2D(3, (3, 3), activation="sigmoid", kernel_initializer='he_normal',padding="same")(x)

        #Compile model
        self.model = tf.keras.Model(input,x)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.01),
            loss=tf.keras.losses.MeanSquaredError(),
            )
        
        #Reload model weights (if they exist)
        try:
            self.model.load_weights('model').expect_partial()
        except:
            print('> WARNING: saved model weights could not be found/restored')
        return

    def load_training_images(self,dir=TRAIN_ROOT,max_N=100):
        #Load training images to member data. Assigns noisy images as x-input and
        #ground truth images as y-label.
        #Inputs:    dir - image directory to pull from. Level above ground/noisy folders
        #           max_N - maximum (whole) images to include in training set

        #Find training images. If limited to max_N, pick random max_N samples from set
        files = os.listdir(dir + GROUND)
        if len(files) >= max_N: files = rd.sample(files, max_N)

        #Load images from file
        print('Loading and segmenting training images:')
        nonce = True
        for i in tqdm(range(len(files))):
            #Fetch ground image. Before saving, normalize and split into patch images
            #Resultant training set is set of image patches, not whole images
            ground_path = dir + GROUND + files[i]
            ground = v_normalize(get_image(ground_path))
            g_patches = gen_patches(ground,8)

            #Fetch noisy image and process. Corresponsind noisy image to ground image is
            #linked via filename. All pairs share same name w/ noisy image appending "_noise"
            #to image name
            noisy_path = dir + NOISY + files[i].replace('.jpg', '_noise.jpg')
            noisy = v_normalize(get_image(noisy_path))
            n_patches = gen_patches(noisy,8)

            #Append patch images to training sets as member data
            if nonce:
                #Set first image directly to member data, then concatenate all others
                #TODO - likely a better way to do this.
                self.train_x = n_patches
                self.train_y = g_patches
                nonce = False
            else:
                self.train_x = np.concatenate((self.train_x,n_patches))
                self.train_y = np.concatenate((self.train_y,g_patches))
        return
        
    def train(self):
        #Load training images from file
        self.load_training_images()

        #Traing model. Uses noisy images as x-input, and ground truth as y-label
        print("Training model:")
        self.model.fit(
            x=self.train_x,
            y=self.train_y,
            epochs=10,
            batch_size=128,
            shuffle=True,
            )
        
        #Save model weights for future re-use
        self.model.save_weights('model')    
        return
    
    def denoise(self,image):
        #Denoise input image using trained NN

        #Normalize image and split into patches
        image = v_normalize(image)
        patches = gen_patches(image,8)
        
        #Make prediction from NN
        prediction = self.model.predict(patches)

        #Recombine patch images and denormalize before returning
        denoised = stitch_patches(prediction)
        return v_denormalize(denoised)
    
        
#Main ---------------------------------------------------------------------------------------------
def main():
    #TODO - structure into final performance implemenation + analysis (with metrics). Below
    # as initial setup to invoke current content

    #Fetch random test image
    files = os.listdir(TEST_ROOT+NOISY)
    file = rd.choice(files)
    noisy = get_image(TEST_ROOT+NOISY+file)
    ground = get_image(TEST_ROOT+GROUND+file.replace("_noise.jpg", ".jpg"))
    
    #NLM filtering
    filtered_nlm = NLM(noisy)

    #Autoencoder filtering
    auto = Autoencoder()
    auto.train()   #If commented out, Autoencoder() will pull from previously trained weights
    filtered_ae = auto.denoise(noisy)
    display_comparison([ground,noisy,filtered_nlm,filtered_ae])

    return

#Execute Main ===========================================================================================
if __name__ == "__main__":
    main()
