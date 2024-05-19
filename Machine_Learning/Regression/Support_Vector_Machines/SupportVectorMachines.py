#!/usr/bin/env python
# coding: utf-8

# # SVM in Python for Classification Problem using Sklearn - linear

# ## Use the sklearn module to create Linear model.

# In[1]:


import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlxtend


# ## Load the data into Python enviorment (Iris)

# In[2]:


from sklearn.datasets import load_iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names) #Input
y = iris.target #Output


# ## Split the Data

# In[3]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# ## Train the Model

# In[4]:


from sklearn import svm
# Design the model
model = svm.SVC(kernel='linear') #SVM Linear 
model.fit(X_train,y_train)


# ## Test the Model

# In[5]:


from sklearn.metrics import confusion_matrix
# Test result
y_predict = model.predict(X_test)
confusion_matrix(y_test, y_predict) #No mistake


# ## Capture the Accuracy of the Model

# In[6]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predict)*100


# ## Get the Number of Support vectors for Each Class

# In[7]:


model.n_support_


# # SVM in Python for Classification using Sklearn - ploy

# ## Fit the model

# In[8]:


clf = svm.SVC(kernel='poly', degree=3) #degree = 3 for the number of factors
clf.fit(X_train, y_train)


# ## Test the model.

# In[9]:


y_predict = clf.predict(X_test)
confusion_matrix(y_test, y_predict)


# In[10]:


accuracy_score(y_test, y_predict)*100


# # SVM in Python for Classification using sklearn - rbf

# ## Train the model.

# In[11]:


clf = svm.SVC(kernel='rbf', gamma=.3) #gamma = .01* number of factor
clf.fit(X_train, y_train)


# ## Test the model.

# In[12]:


y_predict = clf.predict(X_test)
confusion_matrix(y_test, y_predict)


# In[13]:


accuracy_score(y_test, y_predict)*100


# # Grid Search SVM Methods using sklearn and GridSearchCV

# In[18]:


from sklearn.model_selection import GridSearchCV
parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':[0.1, 0.5, 1, 10], 'degree':[2,3,4]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(X_train, y_train)


# # Select the Best Method and Parameters

# In[19]:


clf.best_params_ #Best models. Parameters should be used to retrain "poly" model (the best model)

