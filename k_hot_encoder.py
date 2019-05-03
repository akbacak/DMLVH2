# https://medium.com/@michaeldelsole/what-is-one-hot-encoding-and-how-to-do-it-f0ae272f1179

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd




dataset = pd.read_csv('/home/ubuntu/keras/enver/dmlvh2/data.csv')
Y = dataset.iloc[:,1].values
mlb = MultiLabelBinarizer()

YY = mlb.fit_transform(Y)
print(YY)
np.savetxt('Y.csv',YY, fmt='%d')

