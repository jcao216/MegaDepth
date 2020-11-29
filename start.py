import os
import torch
import sys
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage import io
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

import tensorflow as tf
encoding_stride = 1

decoding_stride = 2
numberOfChannels = 1

setup_encoder = torch.nn.Conv2d(in_channels=numberOfChannels, out_channels=numberOfChannels, stride=encoding_stride, kernel_size=1)
setup_decoder = torch.nn.ConvTranspose2d(in_channels=numberOfChannels,out_channels=1, stride=decoding_stride,kernel_size=1)

batch_size = 4
def autoencoderV3():
    #downsample image first (encoding)
    img = Input(shape=(384,384,1))
    layer1 = layers.Conv2D(16,(3,3), activation='relu', padding='same',kernel_initializer='he_normal')(img)
    layer2 = layers.MaxPooling2D((2,2),padding='same')(layer1)
    layer3 = layers.Conv2D(8,(3,3), activation='relu', padding='same')(layer2)
    layer4 = layers.MaxPooling2D((2,2),padding='same')(layer3)  
    layer5 = layers.Conv2D(8,(3,3), activation='relu', padding='same')(layer4)
    layer6 = layers.MaxPooling2D((2,2),padding='same')(layer5)
    ##reached inner most layer of AE (latent layer/bottleneck)/end of encoder
    layer7 = layers.Conv2D(8,(3,3), activation='relu', padding='same')(layer6)
    layer8 = layers.UpSampling2D((2,2))(layer7)

    layer8a = layers.add([layer8,layer5])

    layer9 = layers.Conv2D(8,(3,3), activation='relu', padding='same')(layer8a)
    layer10 = layers.UpSampling2D((2,2))(layer9)

    layer10a = layers.add([layer10, layer3])

    layer11 = layers.Conv2D(16,(3,3), activation='relu', padding='same')(layer10a)
    layer12 = layers.UpSampling2D((2,2))(layer11)

    fin_decoded_img = layers.Conv2D(1,(3,3),activation='sigmoid', padding='same')(layer12)
    autoencoder = Model(inputs=(img), outputs=(fin_decoded_img))
    return autoencoder



def img_data(test_img_dir):
    img_arr = np.array([asarray(Image.open(test_img_dir + img)) for img in os.listdir(test_img_dir)])
    
    img_arr.astype('float32')
    new_arr = [arr/255.0 for arr in img_arr] # normalize array between 0 and 1
    new_arr_train, new_arr_test = train_test_split(new_arr, random_state=42, shuffle=False,stratify=None)
    print(">>> Returned test and train set<<<")
    return new_arr_train,new_arr_test, new_arr

def form_dataset_from_imgs(directry,lbls,subset_type):
    return tf.keras.preprocessing.image_dataset_from_directory(directry, 
            labels="inferred",label_mode=None, class_names=None,
            color_mode="grayscale",batch_size=batch_size, image_size=(384,384), 
            shuffle=False,seed=123,validation_split=0.25,subset=subset_type)
"""         return tf.keras.preprocessing.image_dataset_from_directory(directry, 
                labels="inferred",label_mode=None, class_names=None,
                color_mode="grayscale",batch_size=32, image_size=(384,384), 
                shuffle=False,seed=None,validation_split=0.25,subset="training") 
"""



# train_set, test_set, raw_arr = img_data("interpolated_frames/") ##make sure to include '/' to signify directory
# print("train set type: {}; test set type: {} ".format(len(train_set), len(test_set)))
new_dir = 'final_frames/'

##start preparing Dataset type from interpolated frames' depth maps
datagenType = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.0, 
        height_shift_range=0.0,horizontal_flip=False,vertical_flip=False,rescale=1/255,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode='nearest',
        cval=0.0,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0,
        dtype='float'
        )
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.0,
    zoom_range=0.0,
    horizontal_flip=False)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

training_datagen = train_datagen.flow_from_directory('final_frames_modified/class_b')
validation_datagen = test_datagen.flow_from_directory(
    'final_frames_modified/class_a',
    target_size=(384,384),
    batch_size=batch_size,
    class_mode='binary')





tensor_dataset_train = form_dataset_from_imgs('final_frames_modified/', os.listdir(new_dir),"training")
tensor_dataset_validate = form_dataset_from_imgs('final_frames_modified/',os.listdir(new_dir),"validation")
# tensor_dataset_train_np = np.stack(list(tensor_dataset_train))
# tensor_dataset_validate_np = np.stack(list(tensor_dataset_validate))
print((training_datagen))
#print((tensor_dataset_validate_np))


##compile autoencoder function
autoencoder = autoencoderV3()
print(autoencoder.summary())
autoencoder.compile(optimizer='adam', loss = 'mean_squared_error', metrics=['accuracy']) 

#autoencoder.fit(training_datagen,steps_per_epoch=200,epochs=55,validation_data=validation_datagen,validation_steps=800)

print("END OF TEST<<<<<<<<<<<<<<<<<<<<<<<<<<")
##define gradient function
#loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
#optim = keras.optimizers.SGD(learning_rate=1e-3)

# for train_set, valid_set in enumerate(tensor_dataset_train):

#with tf.GradientTape() as tape:
#    logits = autoencoder(tensor_dataset_train,training=True)
        # loss_val = loss_fn(valid_set,logits)
        # reconstr = autoencoder()
        # gradients = tape.gradient(loss_val, autoencoder.trainable_weights)
        # optim.apply_gradients(zip(gradients, model.trainable_weights))
        #if step % 200 == 0:
        #    print("Training loss for one batch at step {}: {}".format(step, float(loss_val)))
##train autoencoder





epochs = 55
print(tensor_dataset_train)
combined_dataset = tf.data.Dataset.zip((tensor_dataset_train,tensor_dataset_validate))
# autoencoder.fit(tensor_dataset_train,tensor_dataset_validate,epochs=epochs)
autoencoder.fit(combined_dataset,epochs=epochs)


#autoencode("demo.png","interpolated_frames/00001000.png")

