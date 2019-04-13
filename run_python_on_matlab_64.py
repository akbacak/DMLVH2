
# First extract the video frames

#coding=utf-8
import os
import math
import cv2
import re
import keras
import numpy as np
from keras.applications import VGG16
from keras.models import load_model
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential,Input,Model,InputLayer
from keras.models import model_from_json
from keras.models import load_model
import numpy as np    # for mathematical operations



# First extract frames

listing = os.listdir("/home/ubuntu/Desktop/Thesis_Follow_Up_3/dmqvRetrieval/Python/qVideos/")

count = 1
for file in listing:
    video = cv2.VideoCapture("/home/ubuntu/Desktop/Thesis_Follow_Up_3/dmqvRetrieval/Python/qVideos/" + file)
    print(video.isOpened())
    framerate = video.get(5)
    os.makedirs("/home/ubuntu/Desktop/Thesis_Follow_Up_3/dmqvRetrieval/Python/qFrames/" + file )
    while (video.isOpened()):
        frameId = video.get(1)
        success,image = video.read()
        #if( image != None ):
        #    image=cv2.resize(image,(224,224), interpolation = cv2.INTER_AREA)
        if (success != True):
            break
        if (frameId % math.floor(framerate) == 0):
            filename = "/home/ubuntu/Desktop/Thesis_Follow_Up_3/dmqvRetrieval/Python/qFrames/" + file +"/" + file + "_" + str(int(frameId / math.floor(framerate))+1) + ".jpg"
            print(filename)
            cv2.imwrite(filename,image)
    video.release()
    print('done')
    count+=1



# Second, resize images to 224x224x3 for vgg-16
os.system('sh /home/ubuntu/Desktop/Thesis_Follow_Up_3/dmqvRetrieval/Python/resize.sh')







# Third preprocess the video frames from VGG-16



listing = os.listdir("/home/ubuntu/Desktop/Thesis_Follow_Up_3/dmqvRetrieval/Python/qFrames/")
for file in listing:
    listing_2  = os.listdir("/home/ubuntu/Desktop/Thesis_Follow_Up_3/dmqvRetrieval/Python/qFrames/" + file + "/" )

    X = []
    for images in listing_2:
        image =  plt.imread("/home/ubuntu/Desktop/Thesis_Follow_Up_3/dmqvRetrieval/Python/qFrames/" + file + "/" + images )
        X.append (image)
    X = np.array(X)
    print(X.shape)

    image_size=224,
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(X.shape[1:]))

    #batch_size = 16
    #XX = base_model.predict(X, batch_size=batch_size, verbose=0, steps=None)
    XX = base_model.predict(X, verbose=0, steps=None)
    np.save(open("/home/ubuntu/Desktop/Thesis_Follow_Up_3/dmqvRetrieval/Python/preprocessed/" + file + ".npy", 'w'), XX)



# Fourth, generate hash code like video features from HEL layer

np.set_printoptions(linewidth=8192)

json_file = open('/home/ubuntu/keras/enver/dmlvh/models/dmlvh_v2_no_lstm_64_model.json', 'r')
model = json_file.read()
json_file.close()
model = model_from_json(model)
model.load_weights("/home/ubuntu/keras/enver/dmlvh/models/dmlvh_v2_no_lstm_64_weights.h5")
model = Model(inputs=model.input, outputs=model.get_layer('dense_2').output) # dense_2 for features , dense_3 for predictions


listing = os.listdir("/home/ubuntu/Desktop/Thesis_Follow_Up_3/dmqvRetrieval/Python/preprocessed/")
for file in listing:
    X = []
    X = np.load(open("/home/ubuntu/Desktop/Thesis_Follow_Up_3/dmqvRetrieval/Python/preprocessed/" + file))
    X.shape
   # X = X.reshape(X.shape[0], X.shape[1] * X.shape[2], X.shape[3])
    features = model.predict(X, batch_size=64, verbose=0, steps=None)
    #features = features > 0.5
    features = features.astype(float)
    np.savetxt("/home/ubuntu/Desktop/Thesis_Follow_Up_3/dmqvRetrieval/Python/qFeatures/" + file + ".txt", features, fmt='%f')


# Fifth Generate query labels
np.set_printoptions(linewidth=8192)

json_file = open('/home/ubuntu/keras/enver/dmlvh/models/dmlvh_v2_no_lstm_64_model.json', 'r')
model_p = json_file.read()
json_file.close()
model_p = model_from_json(model_p)
model_p.load_weights("/home/ubuntu/keras/enver/dmlvh/models/dmlvh_v2_no_lstm_64_weights.h5")
model_p = Model(inputs=model_p.input, outputs=model_p.get_layer('dense_3').output) # dense_2 for features , dense_3 for predictions


for file in listing:
    X = []
    X = np.load(open("/home/ubuntu/Desktop/Thesis_Follow_Up_3/dmqvRetrieval/Python/preprocessed/" + file))
    X.shape
   # X = X.reshape(X.shape[0], X.shape[1] * X.shape[2], X.shape[3])
    labels = model_p.predict(X, batch_size=64, verbose=0, steps=None)
    labels = labels > 0.5
    labels = labels.astype(int)
    np.savetxt("/home/ubuntu/Desktop/Thesis_Follow_Up_3/dmqvRetrieval/Python/qLabels/" +"label_" + file + ".txt", labels, fmt='%d')


