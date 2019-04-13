import cv2
import keras
import numpy as np
from keras.applications import VGG16
from keras.models import load_model
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize   # for resizing images



image_size=224
batch_size=64
image_1 = plt.imread("/home/ubuntu/Desktop/Thesis_Follow_Up_3/Datasets/CPSM/KSBY_2009_06_03/0022.jpg") 
image_1 = np.array(image_1)    # converting list to array
image_1 = resize(image_1, preserve_range=True, output_shape=(224,224)).astype(int)      # reshaping to 224*224*3
image_1 = np.array(image_1)

from keras.applications.vgg16 import preprocess_input
image_1 = preprocess_input(image_1, mode='tf')      # preprocessing the input data
image_1 = np.reshape(image_1, (1, 224, 224, 3))
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
image_1 = base_model.predict(image_1, batch_size=batch_size, verbose=0, steps=None)
image_1.shape
image_1 = image_1.reshape(image_1.shape[0], image_1.shape[1] * image_1.shape[2], image_1.shape[3])



image_2 = plt.imread("/home/ubuntu/Desktop/Thesis_Follow_Up_3/Datasets/CPSM/KSBY_2009_06_03/0077.jpg")
image_2 = np.array(image_2)    # converting list to array
image_2 = resize(image_2, preserve_range=True, output_shape=(224,224)).astype(int)      # reshaping to 224*224*3
image_2 = np.array(image_2)
image_2 = preprocess_input(image_2, mode='tf')      # preprocessing the input data
image_2 = np.reshape(image_2, (1, 224, 224, 3))
image_2 = base_model.predict(image_2, batch_size=batch_size, verbose=0, steps=None)
image_2.shape
image_2 = image_2.reshape(image_2.shape[0], image_2.shape[1] * image_2.shape[2], image_2.shape[3])




#coding=utf-8
from keras.models import Sequential,Input,Model,InputLayer
from keras.models import model_from_json
from keras.models import load_model
import numpy as np    # for mathematical operations

np.set_printoptions(linewidth=1024)


json_file = open('models/dmlvh_v2_64_model.json', 'r')
model = json_file.read()
json_file.close()
model = model_from_json(model)
model.load_weights("models/dmlvh_v2_64_weights.h5")
model = Model(inputs=model.input, outputs=model.get_layer('lstm_1').output)########################## # CHANGED TO LSTM_1


features_1 = model.predict(image_1, batch_size=64, verbose=0, steps=None)[0] #If we  do not use [0], output will be [[ ... ]] 
features_1 = features_1 > 0.5
binary_codes_1 = features_1.astype(int)
print(binary_codes_1)


features_2 = model.predict(image_2, batch_size=64, verbose=0, steps=None)[0] #If we  do not use [0], output will be [[ ... ]]
features_2 = features_2 > 0.5
binary_codes_2 = features_2.astype(int)
print(binary_codes_2)

hamming_dis = np.count_nonzero(binary_codes_1 != binary_codes_2)
print "hamming distance: %d" % hamming_dis

