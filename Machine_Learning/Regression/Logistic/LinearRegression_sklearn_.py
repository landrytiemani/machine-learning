#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression in Python using Sklearn

# # Use the sklearn module to create LR model.

# In[4]:


import sklearn
import numpy as np
import pandas as pd


# # Load the data into Python enviorment

# In[5]:


from sklearn.datasets import load_iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target


# # Split the Data

# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# # Train the Model

# In[7]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs',
multi_class ='multinomial') #Since it is a multiclass
model.fit(X_train,y_train)


# # Test the Model

# In[8]:


from sklearn.metrics import confusion_matrix
# Test result
y_predict = model.predict(X_test)
confusion_matrix(y_test, y_predict) #Just  error done on prediction


# # Capture the Accuracy of the Model

# In[9]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predict)*100 #Good accuracy on this test set


# # Capture the Intercept and Coefficients

# In[10]:


model.intercept_, model.coef_

