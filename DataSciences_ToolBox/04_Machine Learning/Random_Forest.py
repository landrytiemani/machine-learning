# Title: Random Forest Classification in Python

# Purpose: This script demonstrates how to train and evaluate a Random Forest classifier on the Iris dataset, and visualize feature importances.

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Split the data into training (80%) and validation (20%) sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=7)

# Train the Random Forest classifier
clf = RandomForestClassifier(random_state=7)

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

# Feature Importance using permutation_importance (similar to varImp in R)
result = permutation_importance(clf, X_valid, y_valid, n_repeats=10, random_state=42)
feature_importances = pd.DataFrame({'feature': iris.feature_names, 'importance': result.importances_mean})

# Sort by importance
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importances, color='skyblue')
plt.title("Feature Importances (Random Forest)")
plt.xlabel("Importance (Mean Decrease in Accuracy)")
plt.show()
