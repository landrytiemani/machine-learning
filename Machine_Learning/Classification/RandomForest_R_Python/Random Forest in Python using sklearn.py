# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 16:29:13 2022

@author: owner
"""

#Use the sklearn module to create random forest models.
import sklearn
import numpy as np
import pandas as pd

#Load the data into Python enviorment
from sklearn.datasets import load_iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names) #Input
y = iris.target #Output

#Split the Data (random split)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#Train the Model
from sklearn.ensemble import RandomForestClassifier #RFClassifier just handle classification problem In contrary by changing the controls, carret can handle classification and regression
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

#Test the Model
from sklearn.metrics import confusion_matrix
# Test result
y_predict = clf.predict(X_test)
species = np.array(y_test).argmax(axis=1) #setup your known (y_test)
predictions = np.array(y_predict).argmax(axis=1) #setup predicted y_pred
confusion_matrix(species, predictions) #Build confusion matrix
#The confusion matrix shows error out of the diagonal. In this case, we made 1 prediction error

#Capture the Accuracy of the Model
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predict)*100
#Accuracy 97.36

#Check the Most Important Features
import pandas as pd
clf.feature_importances_
#Petal Length and petal width are the most important features of the model. Can be used to build a new model
