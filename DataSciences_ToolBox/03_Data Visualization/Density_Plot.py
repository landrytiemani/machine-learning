# Title: Density Plots for Feature Visualization

# Purpose: This script creates density plots to visualize the distribution of features (Sepal Length, Sepal Width, Petal Length, Petal Width) across different species in the Iris dataset.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['Species'] = iris.target_names[iris.target]  # Add species labels

# Set up the figure with subplots
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 12))

# Create density plots for each feature
for i, feature in enumerate(iris.feature_names):
    sns.kdeplot(
        data=df, 
        x=feature, 
        hue="Species", 
        common_norm=False,  # Ensure each plot has its own density scale
        fill=True, 
        alpha=0.7,  # Add transparency for better visualization
        ax=axes[i]  # Plot on the corresponding subplot
    )

    axes[i].set_title(f"Density Plot of {feature}")

plt.tight_layout()  # Adjust spacing between subplots
plt.show()
