# Title: Comprehensive Exploratory Data Analysis (EDA) of the Iris Dataset

# Purpose: This script performs an in-depth analysis of the Iris dataset, including summary statistics, visualizations (scatter plots, histograms, bar plots, density plots, box plots, and pair plots), outlier imputation, and data normalization.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from pandas.api.types import is_numeric_dtype
from scipy.stats import skew, kurtosis
from matplotlib import gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['Species'] = iris.target_names[iris.target]
df_original = df.copy()

# Function to generate summary statistics
def extended_describe(df):
    stats = df.describe()
    for col in stats.columns:
        if is_numeric_dtype(df[col]):
            stats.loc['skew', col] = skew(df[col].dropna())
            stats.loc['kurtosis', col] = kurtosis(df[col].dropna())
    return stats

# 3. Explore Original Data

# Summary Statistics
print("\nSummary Statistics of Original Data:")
print(extended_describe(df).to_markdown(numalign="left", stralign="left"))

# Correlation Matrix and Plot
correlation_matrix = df.iloc[:, :-1].corr() # exclude the last column as it is non-numeric
print("\nCorrelation Matrix:")
print(correlation_matrix.to_markdown(numalign="left", stralign="left"))

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Correlation Matrix for Iris Dataset')
plt.savefig('pictures/cor.png')
plt.show()

# Visualizations
def create_scatterplot(data, x_col, y_col, hue=None, title=None):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue)
    sns.regplot(data=data, x=x_col, y=y_col, scatter=False, ax=plt.gca())  
    sns.despine()
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    if title:
        plt.title(title, fontsize=14)

create_scatterplot(df, 'Sepal.Width', 'Sepal.Length', hue='Species', title='Scatterplot of Sepal Width vs. Sepal Length by Species')
plt.savefig("pictures/mainscatter.png")
plt.show()



# Optional Scatter Plots
create_scatterplot(df, 'Sepal.Width', 'Petal.Width', hue='Species', title='Scatterplot of Sepal Width vs. Petal Width by Species')
plt.savefig("pictures/optional_scatters_1.png")
plt.show()

# Pair Plot
sns.pairplot(df, hue='Species', diag_kind='kde')
plt.suptitle('Pair Plot of Iris Dataset')
plt.savefig("pictures/scattermatrix.png")
plt.show()

# Pair Plot (Alternative)
sns.pairplot(df, hue='Species', diag_kind='hist')
plt.suptitle('Pair Plot of Iris Dataset (Alternative)')
plt.savefig("pictures/optional_scatter_matrix.png")
plt.show()


# Histograms
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for i, feature in enumerate(['Sepal.Width', 'Petal.Width', 'Sepal.Length', 'Petal.Length']):
    row = i // 2
    col = i % 2
    sns.histplot(data=df, x=feature, hue='Species', ax=axes[row, col])
    axes[row, col].set_title(f"Histogram of {feature} by Species")

plt.tight_layout()
plt.show()



# Bar Plots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for i, feature in enumerate(['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']):
    row = i // 2
    col = i % 2
    sns.barplot(data=df, x='Species', y=feature, ax=axes[row, col], errorbar=('ci', 95))
    axes[row, col].set_title(f'Mean {feature} by Species with 95% Confidence Intervals')
    sns.despine()


# Box Plots
fig, axes = plt.subplots(1, 4, figsize=(15, 5))

for i, feature in enumerate(df.columns[:-1]):  
    sns.boxplot(x='Species', y=feature, data=df, ax=axes[i])
    axes[i].set_title(f'Boxplot of {feature} by Species')
plt.show()


# Density Plots
fig, axes = plt.subplots(4, 1, figsize=(8, 12))
for i, feature in enumerate(df.columns[:-1]):  # Assuming 'Species' is the last column
    sns.kdeplot(data=df, x=feature, hue="Species", common_norm=False, fill=True, alpha=0.7, ax=axes[i])
    axes[i].set_title(f"Density Plot of {feature}")
plt.tight_layout()
plt.show()

# Line Plot
plt.figure(figsize=(10, 6))
sns.lineplot(x='Species', y='Sepal.Length', data=df, marker='o', color="blue", errorbar=('ci', 95))
sns.lineplot(x='Species', y='Sepal.Width', data=df, marker='o', color="red", errorbar=('ci', 95))
plt.title('Mean Sepal.Length and Sepal.Width by Species with 95% Confidence Intervals', fontsize=14)
plt.xlabel('Species', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.grid(axis='y', alpha=0.5)
plt.show()


# Preprocessing: Outlier Imputation and Normalization
# ... (same as in the previous response, using `LocalOutlierFactor` and `StandardScaler`)


# Visualize the Processed Data
# ... (plots for processed data are similar to those above, just use the `scaled_df` instead of `df`)

#Save the output plot to PDF
#def save_plots_to_pdf(plots, filename):
#    with PdfPages(filename) as pdf:
#        for plot in plots:
#            fig = plot.get_figure()
#            canvas = FigureCanvas(fig)
#            canvas.print_figure(pdf)
#
#plots = [p, t, g, tt, p1, p2, p3, p4]  # Add all the plot objects to this list
#save_plots_to_pdf(plots, "grid.pdf") 



