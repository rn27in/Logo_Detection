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

def resize(img,width = None, height = None, inter = cv2.INTER_AREA ):
    dim = None
    (h,w) = img.shape[:2]
    if width is None and height is None:
        return img
    if width is None:
        r = height/(float(h))
        dim = (int(r*w), height)
    else:
        r = width/float(w)
        dim = (width, int(r*h))
    resized = cv2.resize(img, dim, interpolation= inter)

    return resized

def resize_image(list_images,path, outpath, width ):
    for img_temp in list_images:
        img = cv2.imread(path +img_temp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = resize(img, width= width)
        cv2.imwrite(outpath+ img_temp, img)

def image_generator(input, input_path, output_path):
    '''

    :param input: list of filenames 
    :param input_path: Path to input files
    :param output_path: path where augmented images are stored
    :return: Does not return anythng. Files are saved to disk automatically
    '''
    gen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.2,
        height_shift_range=0.2,
        # rescale=1. / 255,
        shear_range=0.1
    )

    count = 30
    for image in input:
        temp_image = img_to_array(load_img(input_path + image))
        temp_image = temp_image.reshape((1,) + temp_image.shape)
        images_flow = gen.flow(temp_image, batch_size=32, save_to_dir=output_path)
        for i, new_images in enumerate(images_flow):
            if i >= count:
                break

def import_images_in_array(list_train_pics,path,shape):
    image_list = []
    for img_name in list_train_pics:
            temp_img = image.load_img(path + img_name, target_size= shape)
            temp_img = image.img_to_array(temp_img)
            #temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
            temp_img = temp_img.astype('uint8')
            image_list.append(temp_img)

    return(image_list)


def create_convnet():
    ##Start Layer 1 architecture################
    input_shape = Input(shape=(50, 50, 1), name = "Input_Params")  ###Data##########
    conv1 = ZeroPadding2D((3,3), name = "Pad_for_Conv1")(input_shape) ####Zero Padding of 3##
    ###Shape = 112x112x64###########
    conv1 = Conv2D(64, kernel_size = (7, 7), strides=(2, 2), activation='relu', name = "Conv1")(conv1) ##Convolution 1######
    ####Shape below = 56x56x64#########
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name = "Pool1")(conv1)####Pooling 1############
    ######Shape below = 56x56x64##########
    batchnorm1 = BatchNormalization(name = "Batchnorm1")(pool1)  ###Batch Normalization 1 Instead of LRN##############
    #####Entering the FeatX module now###################
    batchnorm2 = batchnorm1  ##We need to create a paralle stream first################ Hence taking a copy ########
    ##Shape = 56x56x96 ####
    conv2a = Conv2D(96, kernel_size = (1, 1), strides=(1,1), activation='relu', name = "Conv2a")(batchnorm1) ##Build Convlution 2a########
    conv2a = ZeroPadding2D((1, 1), name = "Pad_for_Conv2a")(conv2a)  ##Zero Padding of 1 for building Convolution 2b####
    ###Shape = 56x56x208#############
    conv2b = Conv2D(208, kernel_size=(3, 3), strides=(1, 1), activation='relu', name = "Conv2b")(conv2a) ###Building Convolution 2b now ####
    ###Building the Pooling Layer in Parallel for the first module of FEATX#########
    batchnorm2 = ZeroPadding2D((1, 1), name = "Pad_for_Pool2a")(batchnorm2)  ####Zero Padding of 1## This is for Pooling 2a Layer##
    ###Shape below = 56x56x64########
    pool2a = MaxPooling2D((3,3), strides=(1,1), name = "Pool2a")(batchnorm2)####Pooling 2a Layer############
    ###Shape below = 56x56x64########
    conv2c = Conv2D(64, kernel_size = (1,1), strides = (1,1),activation='relu', name = "Conv3c")(pool2a)
    ###https://stackoverflow.com/questions/43196636/how-to-concatenate-two-layers-in-keras###
    ###Shape below = 56x56x272########
    concat2 = concatenate([conv2b, conv2c], name = "Concat2")##Merging Convolution 2b and 2c##### This is the Concat 2 Layer####
    ###Shape below = 28x28x272########
    pool2b = MaxPooling2D((2,2),strides=(2,2), name = "Pool2b")(concat2)  ####Pooling 2b Layer####
    #########
    ##########Entering the second FeatX module now###################
    pool2b_backup = pool2b
    #####Building the convolutional layers now####
    ##Shape below = 28x28x96#########
    conv3a = Conv2D(96, kernel_size = (1, 1), strides=(1, 1), activation='relu', name = "Conv3a")(pool2b) ##Convolution 3a######
    conv3a = ZeroPadding2D((1, 1), name = "Pad_for_Conv3b")(conv3a)  ####Zero Padding of 3##
    ###Shape below = 28x28x208#########
    conv3b = Conv2D(208, kernel_size= (3,3), strides = (1,1), name = "Conv3b")(conv3a) ###Convolution 3b#######
    pool2b_backup = ZeroPadding2D((1,1), name = "Pad_for_Pool3a")(pool2b_backup)
    ###Shape below = 28x28x272####
    pool3a = MaxPooling2D((3,3), strides = (1,1), name = "Pool3a")(pool2b_backup)  ###Pooling 3a######
    ###Shape below = 28x28x64###########
    conv3c = Conv2D(filters = 64, kernel_size = (1,1), strides = (1,1), name = "Pool3c")(pool3a)  ######Convolution 3c######
    ###Shape below = 28x28x272########
    concat3 = concatenate([conv3b, conv3c], name = "Concat3")  ###Concat ###############
    ###Shape below = 14x14x272######
    pool3b = MaxPooling2D(pool_size = (2,2), strides = (2,2), name = "Pool3b")(concat3) ####Pooling 3b########
    flatten = Flatten()(pool3b)
    fc1 = Dense(128, activation= "relu", name = "FC1")(flatten)
    fc1 = Dropout(0.3, name = "Dropout")(fc1)
    softmax_layer = Dense(1, activation= "sigmoid", name = "Sigmoid")(fc1)
    model = Model(input_shape, softmax_layer)
    model.compile(optimizer= 'adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

if __name__ == '__main__':

    ##Original path to logos#############
    path = '../Logos/'
    list_images = os.listdir(path)
    outpath_25 = '../outpath_25/'
    outpath_50 = '../outpath_50/'
    resize_image(list_images, path, outpath_25, width = 25)
    resize_image(list_images, path, outpath_50, width=50)

    #Generate new images from Image Generator###############
    gen_images_25_path = '../gen_25_path/'
    list_images = os.listdir(outpath_25)
    image_generator(input=list_images, input_path= outpath_25, output_path=gen_images_25_path)

    gen_images_50_path = '../gen_50_path/'
    list_images = os.listdir(outpath_50)
    image_generator(input=list_images, input_path= outpath_50, output_path=gen_images_50_path)

    neg_path = '../negative/'
    list_images = os.listdir(neg_path)
    neg_path_final = '../neg_resized_25/'

    ###########Resizing negative images with size 25x25#############
    resize_image(list_images, neg_path, neg_path_final, width=25)

    neg_path_final = '../neg_resized_50/'

    ###########Resizing negative images with size 50x50#############
    resize_image(list_images, neg_path, neg_path_final, width=50)

    pos_images = os.listdir(gen_images_50_path)
    neg_images = os.listdir(neg_path_final)

    pos_image_list = import_images_in_array(pos_images, gen_images_50_path, (50, 50))
    neg_image_list = import_images_in_array(neg_images, neg_path_final, (50, 50))

    final_img = pos_image_list + neg_image_list
    final_img = np.array(final_img)
    final_img = final_img / 255.0

    final_labels = np.array(list(np.repeat(1, len(pos_image_list))) + list(np.repeat(0, len(neg_image_list))))
    final_labels = final_labels.reshape(len(final_labels), 1)

    #####Reshaping images for CNN########################
    final_img = np.reshape(final_img, (final_img.shape[0], final_img.shape[1], final_img.shape[2], 1))

    model = create_convnet()

    start = time.time()
    history = model.fit(final_img, final_labels, validation_split=0.1, epochs= 5, verbose= 1)
    end = time.time()

    pred = model.predict(final_img, verbose=1)



