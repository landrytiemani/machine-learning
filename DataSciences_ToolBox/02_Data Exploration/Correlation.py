# Title: Correlation Analysis with Python

# Purpose: This script calculates and displays the correlation matrix for the numerical features of the Iris dataset.

import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Print the correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix.to_markdown(numalign="left", stralign="left"))
