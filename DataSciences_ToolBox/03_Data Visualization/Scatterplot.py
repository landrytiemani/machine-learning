# Title: Scatterplots in Python

# Purpose: This script demonstrates how to create various types of scatterplots:
# 1. Simple scatter plot with linear regression line.
# 2. Grouped scatter plot colored by species with linear regression line.
# 3. Alternative scatter plot using Seaborn's `relplot`.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['Species'] = iris.target_names[iris.target]  # Add species labels


# Simple Scatter Plot with Linear Regression
plt.figure(figsize=(8, 6))  
sns.regplot(x="Sepal.Width", y="Sepal.Length", data=df, scatter_kws={'color': 'black'}, line_kws={'color': 'blue'})

# Customize plot appearance
sns.despine()  # Remove top and right spines
plt.xlabel('Sepal Width', fontsize=12)
plt.ylabel('Sepal Length', fontsize=12)
plt.title('Scatterplot of Sepal Width vs. Sepal Length with Regression Line', fontsize=14)

plt.show()


# Grouped Scatter Plot
plt.figure(figsize=(8, 6))
sns.lmplot(x="Sepal.Width", y="Sepal.Length", data=df, hue="Species")

# Customize plot appearance
plt.xlabel('Sepal Width', fontsize=12)
plt.ylabel('Sepal Length', fontsize=12)
plt.title('Scatterplot of Sepal Width vs. Sepal Length by Species', fontsize=14)

plt.show()

# Optional: Alternative Scatter Plot using relplot
sns.relplot(x="Sepal.Width", y="Sepal.Length", hue="Species", size="Petal.Width", data=df)

plt.show()
