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

dataset = pd.read_csv('/home/ubuntu/Desktop/Thesis_Follow_Up_3/Datasets/CPSM/data2.csv')
Y2 = dataset.iloc[:,1].values
print(Y2)

Y2 =  to_categorical(Y2)
print(Y2)

features = Y2
features = features.astype(int)
np.savetxt('Y2.csv',features, fmt='%d')
