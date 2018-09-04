from keras.layers import Conv2D, MaxPooling2D, Input,Concatenate,  concatenate, Flatten, Dense,Dropout,Input
import os, re,cv2
import numpy as np
from collections import OrderedDict
from keras import optimizers
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.layers import Concatenate,  concatenate
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import  time
from keras.models import Sequential, optimizers


def convnet():
    conv_model = Sequential()
    # add convolutional layer with 16 5x5 filters using relu activation
    conv_model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(50, 50, 1)))
    # add maxpooling layer
    conv_model.add(MaxPooling2D((2, 2)))
    # add convolutional layer with 16 5x5 filters using relu activation
    conv_model.add(Conv2D(16, (5, 5), activation='relu'))
    # add maxpooling layer
    conv_model.add(MaxPooling2D((2, 2)))
    # add convolutional layer with 16 5x5 filters using relu activation
    conv_model.add(Conv2D(16, (5, 5), activation='relu'))
    conv_model.add(MaxPooling2D((2, 2)))
    # add flatten layer to reduce output shape
    conv_model.add(Flatten())
    # add output layer
    conv_model.add(Dense(1, activation='sigmoid'))
    conv_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    return conv_model



