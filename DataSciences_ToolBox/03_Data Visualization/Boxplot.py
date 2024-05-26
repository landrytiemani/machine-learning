# Title: Bar Plots with Error Bars in Python

# Purpose: This script creates a bar plot with error bars to visualize the mean and confidence intervals of sepal length across different Iris species.

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['Species'] = iris.target_names[iris.target]  # Add species labels

# Create bar plot with error bars using Seaborn
plt.figure(figsize=(8, 6))  # Optional: Adjust figure size

sns.barplot(
    x="Species", 
    y="Sepal.Length", 
    data=df, 
    color="white",  # Fill color of the bars
    edgecolor="black"  # Border color of the bars
)

# Add error bars (95% confidence interval by default)
sns.despine(left=True)  # Remove the top and right spines
plt.xlabel('Species', fontsize=12)  # Customize x-axis label
plt.ylabel('Sepal Length', fontsize=12)  # Customize y-axis label
plt.title('Mean Sepal Length by Species with 95% Confidence Intervals', fontsize=14)  # Add title
plt.show()
