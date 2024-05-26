# Title: Histogram in Python

# Purpose: This script creates a histogram to visualize the distribution of Sepal Width in the Iris dataset.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Create the histogram using Seaborn
plt.figure(figsize=(8, 6))  # Optional: Adjust figure size

sns.histplot(df['Sepal.Width'], binwidth=0.4, color='green', kde=False)  # kde=False to remove the density curve
sns.despine()  # Remove the top and right spines for a cleaner look

plt.xlabel("Sepal Width", fontsize=12)  # Customize x-axis label
plt.ylabel("Frequency", fontsize=12)    # Customize y-axis label
plt.title('Distribution of Sepal Width in Iris Dataset', fontsize=14)  # Add title
plt.grid(axis='y', alpha=0.5)  # Add a subtle horizontal grid

plt.show()
