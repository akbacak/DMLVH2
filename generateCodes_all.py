#coding=utf-8
from keras.models import Sequential,Input,Model,InputLayer
from keras.models import model_from_json
from keras.models import load_model
import numpy as np    # for mathematical operations
import os



np.set_printoptions(linewidth=8192)


json_file = open('models/test_64_model.json', 'r')
model = json_file.read()
json_file.close()
model = model_from_json(model)
model.load_weights("models/test_64_weights.h5")
model = Model(inputs=model.input, outputs=model.get_layer('dense_2').output) # dense_2 for features , dense_3 for predictions



'''
listing = os.listdir("/home/ubuntu/keras/enver/dmlvh/preprocessed_videos/")
for file in listing:
    X = []
    X = np.load(open("/home/ubuntu/keras/enver/dmlvh/preprocessed_videos/" + file))
    X.shape
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2], X.shape[3])
    features = model.predict(X, batch_size=64, verbose=0, steps=None)
    #features = features > 0.5
    features = features.astype(float)
    np.savetxt("features/" + file + ".txt", features, fmt='%f') 

'''

X = []
X = np.load(open("NP.npy"))
X.shape
print(X.shape[:])
X = X.reshape(X.shape[0], X.shape[1] , X.shape[2] * X.shape[3] * X.shape[4])
features = model.predict(X, batch_size=64, verbose=0, steps=None)
features = features > 0.5
features = features.astype(int)
np.savetxt("test.txt", features, fmt='%d')

