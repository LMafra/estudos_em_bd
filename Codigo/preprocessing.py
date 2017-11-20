import pandas as pd
from sklearn import preprocessing
import numpy as np
from pandas.tools.plotting import parallel_coordinates


data = pd.read_csv("iris.csv");

data2 = data[['sepal_length','sepal_width','petal_length','petal_width']]

print(data2)

#X = np.array([[ 1., -1.,  2.],[ 2.,  0.,  0.], [ 0.,  1., -1.]])
X_scaled = preprocessing.scale(data2)
print(X_scaled)

X_normalized = preprocessing.normalize(data2)
print(X_normalized)

# MISSING VALUES

from sklearn.preprocessing import Imputer
from sklearn.feature_extraction import DictVectorizer

data3 = pd.read_csv("heart.csv", header=None, na_values=[" "])



data3.replace(' ', np.NaN)

#for j in range(0,len(data.columns)):

print("Medias")
avg = data3.mean()
print(avg)

X = data3.ix[:,0:13]
Y = data3.ix[:,13]


for ind in range(0,len(avg)):
    for line in range(0,len(data3.index)):
        data3[ind].replace(np.nan,avg[ind])

#imputer=Imputer(missing_values=np.NaN, strategy='mean', axis=0)

#imputer = imputer.fit(data3)
#data4 = imputer.transform(data3)

print(data3)

# MAPPING CATEGORIAL ATTRIBUTES TO NUMERICAL ONES

for i in range(0,len(data.columns)):
    print(i)

data_iris = data.copy(deep=True)

labels = data_iris.ix[:, 4]

class_values = labels.unique()

#print(class_values)

#print(len(data_iris.index))

#print(number_of_values)

for line in range(0,len(data_iris.index)):
    for value_index in range(0,len(class_values)):
        #print(data.ix[line,4])
        if data_iris.ix[line,4] == class_values[value_index]:
            data_iris.ix[line,4] = value_index+1

#print(data_iris)

# FEATURE SELECTION

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
array = data.values
X = array[:,0:4]
Y = array[:,4]
print(X)
print(Y)
model.fit(X,Y)
print(model.feature_importances_)

from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=2).fit(X, Y)
x_selected = selector.transform(X)
scores = selector.scores_
print(scores)
print(x_selected)

#sampling

import random as rd
rows = rd.sample(data_iris.index, 10)
print(rows)

#rows2 = np.random.choice(df.index.values, 10)