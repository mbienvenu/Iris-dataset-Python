# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 13:07:18 2018

@author: D'Costa
"""

#iris SVM



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iris = pd.read_csv("Iris_dataset.csv")


#separating the independent variables - pick all rows and all columns except
# the  last one
x=iris.iloc[:,0:4].values # independent variables should always be a matrix 
#the dependent variables
y=iris.iloc[:,4].values

np.set_printoptions(threshold=100) 

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)



from sklearn.cross_validation import train_test_split
#to match the same data in the sets, set random_state to the same number as the trainer
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=1/3,random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)




# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state=0)
classifier.fit(xtrain,ytrain)

# Predicting the Test set results
ypred = classifier.predict(xtest)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, ypred)