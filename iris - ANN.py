# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 17:06:51 2018

@author: D'Costa
"""


#iris ANN



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

# Start ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing the ANN
classifier = Sequential()


#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 2, init = 'uniform', activation ='relu',
                     input_dim = 4))

#Add the second hidden layer
#classifier.add(Dense(output_dim = 3, init = 'uniform', activation ='relu'))

#Add the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation ='sigmoid'))

#Compiling the ANN

classifier.compile(optimizer='adam',loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(xtrain, ytrain, batch_size=10, nb_epoch=100)


# Predicting the Test set results
y_pred = classifier.predict(xtest)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, y_pred)

