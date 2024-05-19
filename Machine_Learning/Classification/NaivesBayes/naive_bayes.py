# Naive Bayes in Python ----
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
from sklearn.model_selection import train_test_split
#Split data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
#Now we have 70% daata training
#EDA
from sklearn.metrics import accuracy_score
gnb = GaussianNB()
gnb = gnb.fit(X_train,y_train)
y_predict = gnb.predict(X_test)
accuracy_score = (y_test,y_predict)*100
print(accuracy_score)

#One way to improve the accuracy
gnb = GaussianNB(priors=[.13,.74,.13])
y_predict = gnb.predict(X_test)
accuracy_score = (y_test,y_predict)*100
print(accuracy_score)

#We can get more info about the model
gnb.__dict__

#Likelihood of features
