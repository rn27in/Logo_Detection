from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop = stopwords.words('english')
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import cv2


def text_preprocessing(text):
    '''

    :param text: raw text from user 
    :return: cleaned text
    '''
    temp = word_tokenize(text)
    cleaned_text = []
    for elem in temp:
        elem = re.sub(r"'s", '', elem)
        elem = elem.lower()
        elem = lemmatizer.lemmatize(elem)
        cleaned_text.append(elem)

    return cleaned_text

def stride_image(image, filter_size, stride_length):
    '''
    
    :param image: image file 
    :param filter_size: filter size
    :param stride_length: stride length
    :return: patches of images
    '''
    h, w = image.shape  ##Obtaning image shape
    ###Determining shape of final array######
    shape = np.array(
        [(((h - filter_size[0]) // stride_length) + 1), (((w - filter_size[1]) // stride_length) + 1), filter_size[1],
         filter_size[0]])
    ###Determining stride shape######
    strides = image.itemsize * np.array([((stride_length * w)), stride_length, w, 1])
    reshaped_array = np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)
    return reshaped_array


#############In case we are using Image Processing###################
######################Main approach#################################################
if __name__ == '__main__':
    ##############Input from user########
    raw_text = raw_input()
    ######Assuming the image path name is provided############
    img_path = 'input_image_from_the_user'
    img = img_to_array(load_img(img_path))

    #####Loading vectorizer and Classifier############
    vectorizer = joblib.load('../vect_text')
    clf = joblib.load('../clf_text')
    lemmatizer = WordNetLemmatizer()

    ##Use the text classifier to predict if we need to give count or "Yes/No"
    cleaned_text = text_preprocessing(raw_text)
    input_matrix = vectorizer.transform(cleaned_text)

    ############Predict whether we need to provide count of images or whether Yes or No########
    pred = clf.predict(input_matrix)
    #####Text ends here######

    #######Logo detection begins here########################
    result = classifier_image(img,pred)
    print result

#######################End here############################################################


if __name__ == '__main__':
    ##############Input from user########
    raw_text = raw_input()
    ######Assuming the image path name is provided############
    img_path = 'input_image_from_the_user'
    image = img_to_array(load_img(img_path))

    #####Loading vectorizer and Classifier############
    vectorizer = joblib.load('../vect_text')
    clf = joblib.load('../clf_text')
    lemmatizer = WordNetLemmatizer()

    ##Use the text classifier to predict if we need to give count or "Yes/No"
    cleaned_text = text_preprocessing(raw_text)
    input_matrix = vectorizer.transform(cleaned_text)
    pred = clf.predict(input_matrix)
    #####Text ends here######

    #######Image classification begins here########################
    img = image.load_img(image, target_size=(256, 256))
    img = image.img_to_array(image)
    ###Create a Gray scale image######################
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #####Create multiple patches of the image######################
    reshaped_array = stride_image(img, (128, 128), 128)
    final_image_patches = []
    for j in range(reshaped_array.shape[0]):
        for k in range(reshaped_array.shape[1]):
            final_image_patches.append(reshaped_array[j][k])
    final_result = []
    for index, img_temp in enumerate(final_image_patches):
        img_temp = img_temp/255.0
        result = classifier_image(img_temp)
        final_result.append(result)
    final_result = np.sum(np.array(final_result))
    if pred == 1:
        print final_result
    else:
        print "Yes" if final_result[0]> 0 else "No"

    ####End######
