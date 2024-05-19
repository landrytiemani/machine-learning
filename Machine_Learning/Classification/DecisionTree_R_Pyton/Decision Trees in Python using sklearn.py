# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 17:55:17 2022
Title: Decision Trees in Python  using sklearn
@author: owner
"""
#pip install pydot
#pip install graphviz
#pip install --upgrade scikit-learn==0.20.3

#Importing Required Libraries
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier #main library
from sklearn.model_selection import train_test_split #main library
from sklearn.metrics import confusion_matrix #main library
from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO 
from six import StringIO
from IPython.display import Image 
from pydot import graph_from_dot_data
import pandas as pd
import numpy as np

#Load the Data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names) #Input
y = pd.Categorical.from_codes(iris.target, iris.target_names) #Output (categorical since this is categorical problem)
X.head()

#Convert Y to Dummy Variables
y = pd.get_dummies(y) #Despite decision tres can handle categorical data, we need to encode in terms of digits (i.e. setosa=0, versicolor=1, virginica=2) in order to create a confusion matrix at a later point

#Split the Data ()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1) #To evaluate the performance of our model, we need to set a quarter of the data aside for testing


#Create the Model
dt = DecisionTreeClassifier() #Create an empty instance of DecisionTreeClassifer class and store it to a variable dt. 
dt.fit(X_train, y_train) #Since this is a supervised machine learning algo, we have to provide y. Then we train the model

#Predicition (how well model performed?)
y_pred = dt.predict(X_test) #load the trained model (dt), use predict function X_test and save it into another variable (y_pred)

#Confusion Matrix
species = np.array(y_test).argmax(axis=1) #setup your known (y_test)
predictions = np.array(y_pred).argmax(axis=1) #setup predicted y_pred
confusion_matrix(species, predictions) #Build confusion matrix
#The confusion matrix shows error out of the diagonal. In this case, we made 1 prediction error

#Calculate the Accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred)) #This model performed at 97%. It iis unrealistic since dataset not big enough

#Structure with Carrett in R or sklearn in python
#1. Get data
#2. Split data
#3. Set controls
#4. Train models
#5. Test models
#5. 
