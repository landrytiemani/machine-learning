# Title: Correlation Plot in Python

# Purpose: This script creates a correlation plot (heatmap) to visualize the relationships between numerical features in the Iris dataset.

import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt


# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Create the correlation plot (heatmap)
plt.figure(figsize=(8, 6))  # Optional: Adjust figure size

sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Correlation Heatmap for Iris Dataset')
plt.show()
