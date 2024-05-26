# Title: Scatterplot Matrix in Python (using Seaborn's PairGrid)

# Purpose: This script creates a scatterplot matrix (pairs plot) to visualize relationships between multiple numerical variables in the Iris dataset.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['Species'] = iris.target_names[iris.target]  # Add species labels

# Create the PairGrid
g = sns.PairGrid(df, hue="Species")

# Map upper triangle to scatter plots
g.map_upper(sns.scatterplot)

# Map lower triangle to density plots
g.map_lower(sns.kdeplot)

# Map diagonal to histograms
g.map_diag(sns.histplot)

# Add a title
g.fig.suptitle("Pair Plot of Iris Dataset", y=1.02)  # Adjust 'y' as needed for title placement

plt.show()


# Purpose: This script creates a scatterplot matrix (pairs plot) to visualize relationships between multiple numerical variables in the Iris dataset.

# Create the pairplot
sns.pairplot(df, hue="Species")

plt.title("Pair Plot of Iris Dataset")  # Add title
plt.show()
