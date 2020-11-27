import os
import torch
import sys
from torch.autograd import Variable
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage import io
import skimage.viewer
from skimage.transform import resize as Resize

import keras
from keras import layers
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Reshape, InputLayer
from keras.datasets import mnist
import matplotlib.pyplot as plt 

import PIL
from PIL import Image
from numpy import asarray

encoding_stride = 1

decoding_stride = 2
numberOfChannels = 1

setup_encoder = torch.nn.Conv2d(in_channels=numberOfChannels, out_channels=numberOfChannels, stride=encoding_stride, kernel_size=1)
setup_decoder = torch.nn.ConvTranspose2d(in_channels=numberOfChannels,out_channels=1, stride=decoding_stride,kernel_size=1)


""" class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init()
        self.convTwoDim = setup_encoder()
        self.convTwoDimTwo = torch.nn.Conv2d()

        self.convTr2d = setup_decoder()
        self.convTr2d2 = torch.nn.ConvTranspose2d()

    def forward(self,x):
        # for encoding
        x = torch.nn.functional.relu(self.convTwoDim(x))
        x = torch.nn.functional.relu(self.convTwoDimTwo(x))

        # for decoding
        x = torch.nn.functional.relu(self.convTr2d(x))
        x = torch.sigmoid(self.convTr2d2(x)) """

""" def autoencode(img, orig_filename):
    ## from vanilla autoencoder
     encoder_dim = 512
    img = Input(shape=(384,512,1)))
    encoded_img = Dense(encoder_dim, activation='relu')(img)
    decoded_img = Dense(384*512, activation='sigmoid')(encoded_img)
    autoencoder = Model(img, decoded_img)

    encoder = Model(img,encoded_img)
    encoded_in = Input(shape=(encoder_dim,))

    decoded_layer = autoencoder.layers[-1]

    decoder = Model(encoded_in, decoder_layer(encoded_in))

    autoencoder.compile(optimizer='adam', loss = 'binary_crossentropy')
    ##-------
    img = Input(shape=(384,512,1)))
    encoded_img = layers.Conv2D(16,(3,3), activation='relu', padding='same')(img)
    encoded_img = layers.MaxPooling2D((2,2),padding='same')(encoded_img)
    encoded_img = layers.Conv2D(8,(3,3), activation='relu', padding='same')(encoded_img)
    encoded_img = layers.MaxPooling2D((2,2),padding='same')(encoded_img)  
    encoded_img = layers.Conv2D(8,(3,3), activation='relu', padding='same')(encoded_img)
    fin_encoded_img = layers.MaxPooling2D((2,2),padding='same')(encoded_img)

    encoded_img = layers.Conv2D(8,(3,3), activation='relu', padding='same')(fin_encoded_img)
    encoded_img = layers.UpSampling2D((2,2))(encoded_img)
    encoded_img = layers.Conv2D(8,(3,3), activation='relu', padding='same')(encoded_img)
    encoded_img = layers.UpSampling2D((2,2))(encoded_img)
    encoded_img = layers.Conv2D(16,(3,3), activation='relu', padding='same')(encoded_img)
    encoded_img = layers.UpSampling2D((2,2),padding='same')(encoded_img)

    fin_decoded_img = layers.Conv2D(1,(3,3),activation='sigmoid', padding='same')(encoded_img)
    autoencoder = Model(img, fin_decoded_img)

    #encoder = Model(img,encoded_img)
    #encoded_in = Input(shape=(encoder_dim,))

    #decoded_layer = autoencoder.layers[-1]
    #decoder = Model(encoded_in, decoder_layer(encoded_in))

    autoencoder.compile(optimizer='adam', loss = 'binary_crossentropy') 

    return fin_encoded_img,fin_decoded_img

 """

def autoencode(img):
    code_size = 512
    encdr = Sequential()
    encdr.add(InputLayer(img))
    encdr.add(Flatten())
    encdr.add(Dense(code_size))

    decdr = Sequential()
    decdr.add(InputLayer((code_size,)))
    decdr.add(Dense(img))

    decdr.add(Reshape(img))

    return encdr,decdr


    ## second iteration---------
"""      img = Input(shape=(384,512,1))
    encoded_img = layers.Conv2D(16,(3,3), activation='relu', padding='same')(img)
    encoded_img = layers.MaxPooling2D((2,2),padding='same')(encoded_img)
    encoded_img = layers.Conv2D(8,(3,3), activation='relu', padding='same')(encoded_img)
    encoded_img = layers.MaxPooling2D((2,2),padding='same')(encoded_img)  
    encoded_img = layers.Conv2D(8,(3,3), activation='relu', padding='same')(encoded_img)
    fin_encoded_img = layers.MaxPooling2D((2,2),padding='same')(encoded_img)

    encoded_img = layers.Conv2D(8,(3,3), activation='relu', padding='same')(fin_encoded_img)
    encoded_img = layers.UpSampling2D((2,2))(encoded_img)
    encoded_img = layers.Conv2D(8,(3,3), activation='relu', padding='same')(encoded_img)
    encoded_img = layers.UpSampling2D((2,2))(encoded_img)
    encoded_img = layers.Conv2D(16,(3,3), activation='relu', padding='same')(encoded_img)
    encoded_img = layers.UpSampling2D((2,2),padding='same')(encoded_img)

    fin_decoded_img = layers.Conv2D(1,(3,3),activation='sigmoid', padding='same')(encoded_img)
    autoencoder = Model(img, fin_decoded_img) """
    # ----------------------


    #encoder = Model(img,encoded_img)
    #encoded_in = Input(shape=(encoder_dim,))

    #decoded_layer = autoencoder.layers[-1]
    #decoder = Model(encoded_in, decoder_layer(encoded_in))

    autoencoder.compile(optimizer='adam', loss = 'binary_crossentropy') 

    # input_img =  torch.from_numpy( np.transpose(img, (2,0,1)) ).contiguous().float()
    # input_img = input_img.unsqueeze(0)    
"""


def img_data(test_img_dir):
    img_arr = np.array([asarray(Image.open(test_img_dir + img)) for img in os.listdir(test_img_dir)])
    
    img_arr.astype(np.float32)
    new_arr = [arr/255.0 for arr in img_arr] # normalize array between 0 and 1
    new_arr_train, new_arr_test = train_test_split(new_arr, random_state=42, shuffle=False,stratify=None)
    print(">>> Returned test and train sets <<<")
    return new_arr_train,new_arr_test, new_arr

# os.system("python demo.py")
train_set, test_set, raw_arr = img_data("interpolated_frames/") ##make sure to include '/' to signify directory
print("train set type: {}; test set type: {} ".format(len(train_set), len(test_set)))
#autoencode("demo.png","interpolated_frames/00001000.png")

""" try:
    os.mkdir("final_frames/")
except OSError:
    print("Directory final frames could not be created")
print(">>> Generating depth frame #1. <<<")

os.rename('interpolated_frames/'+"00001000.png", 'demo.jpg')  ## take the interpolated frame out of its own directory and rename it to demo.jpg so that demo.py does not need to handle different filenames each time
os.system("python demo.py")
os.rename('demo.jpg','interpolated_frames/'+"00001000.png") """
""" file_ct = 1
for filer in os.listdir("interpolated_frames/"):
    print(">>> Generating depth frame #{}. <<<".format(file_ct))
    os.rename('/interpolated_frames'+filer, 'demo.jpg')  ## take the interpolated frame out of its own directory and rename it to demo.jpg so that demo.py does not need to handle different filenames each time
    os.system("python demo.py")
    
    print(">>> Autoencoding depth frame #{}. Please hold. <<<".format(file_ct))
    autoencode("demo.png",filer)
    print(">>> Finished processing depth frame #{}. <<<".format(file_ct))
    file_ct += 1 """