# https://medium.com/@michaeldelsole/what-is-one-hot-encoding-and-how-to-do-it-f0ae272f1179

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
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
one_hot = MultiLabelBinarizer()
Y = dataset.iloc[:,1].values
Y = one_hot.fit_transform(Y)


print(Y)


features = Y
features = features.astype(int)
np.savetxt('Y.csv',features, fmt='%d')
