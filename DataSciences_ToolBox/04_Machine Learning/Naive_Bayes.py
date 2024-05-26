# Title: Naive Bayes Classification in Python

# Purpose: This script demonstrates how to train and evaluate a Naive Bayes classifier on the Iris dataset.

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.naive_bayes import GaussianNB  
from sklearn.metrics import confusion_matrix, classification_report

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Split the data into training (70%) and validation (30%) sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=7)

# Train the Naive Bayes classifier
clf = GaussianNB()  # Gaussian Naive Bayes is suitable for continuous data

# Cross-validation
cv_results = cross_validate(clf, X_train, y_train, cv=10, scoring='accuracy')
print(f"\nCross-Validation Results:\nMean Accuracy: {cv_results['test_score'].mean():.4f}")

# Fit the model
clf.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = clf.predict(X_valid)

# Evaluate the model
confusion_mat = confusion_matrix(y_valid, y_pred)
class_report = classification_report(y_valid, y_pred, target_names=iris.target_names)

print("\nConfusion Matrix:")
print(confusion_mat)
print("\nClassification Report:")
print(class_report)
