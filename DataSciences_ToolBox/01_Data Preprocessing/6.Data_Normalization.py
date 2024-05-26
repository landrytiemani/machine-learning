# Purpose: This script preprocesses the Iris dataset by scaling and centering features, then demonstrates advanced preprocessing techniques for handling missing values, transformations, and feature selection.

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA

# Load and prepare data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target


# Simple Preprocessing: Scale and Center
pipeline = make_pipeline(
    StandardScaler()  # Center and scale features
)
X_scaled_centered = pipeline.fit_transform(X)

print("\nScaled and Centered Data:")
print(X_scaled_centered)



# Advanced Techniques Showcase (with a subset of methods for demonstration)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Example: KNN Imputation (if there were missing values)
imputer = KNNImputer(n_neighbors=5)
X_train_imputed = imputer.fit_transform(X_train)

# Example: PCA (Dimensionality Reduction)
pca = PCA(n_components=2)  # Keep only 2 principal components
X_train_pca = pca.fit_transform(X_train_imputed)


print("\nPCA Transformed Data:")
print(X_train_pca)