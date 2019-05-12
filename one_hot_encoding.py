# https://medium.com/@michaeldelsole/what-is-one-hot-encoding-and-how-to-do-it-f0ae272f1179

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd


'''
dataset = pd.read_csv('data.csv')
Y = dataset.iloc[:,:].values

le = LabelEncoder()

Y[:, 1] = le.fit_transform(Y[:, 1])

ohe = OneHotEncoder(categorical_features = [1])
Y = ohe.fit_transform(Y).toarray()
'''

from keras.utils import to_categorical

dataset = pd.read_csv('/home/ubuntu/keras/enver/dmlvh2/data.csv')
Y = dataset.iloc[:,1].values
print(Y)

YY =  to_categorical(Y)
print(YY)

features = YY
features = features.astype(int)
np.savetxt('Y.csv',features, fmt='%d')
