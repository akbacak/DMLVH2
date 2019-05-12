#coding=utf-8

import cv2
import numpy as np
import sys,os
import time
import matplotlib
import scipy.io
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
from keras import optimizers
import keras
from keras.models import Sequential,Input,Model,InputLayer
from keras.layers import Dense, Dropout, Flatten, TimeDistributed
from keras.layers import Conv2D, MaxPooling2D
from keras.applications import VGG16
from keras.models import model_from_json
from keras.models import load_model
import tensorflow as tf
import pandas as pd  
import matplotlib.pyplot as plt
from skimage.transform import resize   # for resizing images
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.recurrent import LSTM

Y = pd.read_csv(r'/home/ubuntu/keras/enver/dmlvh2/Y.csv') # Video files labels, one hot encoded
Y.shape
print(Y.shape[:])


X = np.load(open('/home/ubuntu/keras/enver/dmlvh2/NP.npy')) # Join video frames in to one NPY file , see deneme.py
X.shape
print(X.shape[:])   # (number of videos, number frames in each video, 7,7,512)
X = X.reshape(X.shape[0], X.shape[1] , X.shape[2] * X.shape[3] * X.shape[4]) # (number of videos, number frames in each video, 7x7x512)
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.25 ,random_state=43)



batch_size = 128
epochs = 300
hash_bits = 512


visible = Input(shape = (X.shape[1] ,X.shape[2]))
lstm    = LSTM(1024 , input_shape=(X.shape[1], X.shape[2]), return_sequences = False )(visible) 
#Dense_1 = Dense(1024, activation='relu')(lstm)
Dense_2 = Dense(hash_bits, activation='sigmoid')(lstm)
Dense_3 = Dense(3, activation='sigmoid')(Dense_2)
model = Model(input = visible, output=Dense_3)
print(model.summary())

import keras.backend as K
# e = 0.5
def c_loss(noise_1, noise_2):
    def loss(y_true, y_pred):
        return (K.binary_crossentropy(y_true, y_pred) + (1/hash_bits) * (K.sum((noise_1 - noise_2)**2) )) 
        #return (K.binary_crossentropy(y_true, y_pred) )
        # change to binary_crossentropy when using multilabel !
    return loss

from keras.optimizers import SGD
sgd = SGD(lr=0.001, decay = 1e-6, momentum=0.9, nesterov=True)
model.compile(loss = c_loss(noise_1 = tf.to_float(Dense_2 > 0.5 ), noise_2 = Dense_2 ),  optimizer=sgd, metrics=['accuracy']) # lstm_1 output is used for optimization
#model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(X, Y, shuffle=True, batch_size=batch_size,epochs=epochs,verbose=1 )
#history = model.fit(X_train, Y_train, shuffle=True, batch_size=batch_size,epochs=epochs,verbose=1, validation_data=(X_valid, Y_valid) )

model_json = model.to_json()
with open("models/dmlvh2_512_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("models/dmlvh2_512_weights.h5")

'''
params = {'legend.fontsize': 20,
          'legend.handlelength': 2,}
plt.rcParams.update(params)


plt.plot(history.history['acc'] , linewidth=3, color="green")
plt.plot(history.history['val_acc'], linewidth=3, color="blue")
plt.title('model accuracy' , fontsize=20)
plt.ylabel('accuracy' , fontsize=20)
plt.xlabel('epoch' , fontsize=20)
plt.legend( ['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'], linewidth=3, color="green")
plt.plot(history.history['val_loss'],  linewidth=3, color="blue")
plt.title('model loss' , fontsize=20)
plt.ylabel('loss' , fontsize=20)
plt.xlabel('epoch' , fontsize=20)
plt.legend( ['train', 'validation'], loc='upper left')
plt.show()
'''
