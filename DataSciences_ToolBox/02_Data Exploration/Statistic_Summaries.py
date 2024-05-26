# Title: Statistical Summaries with Python

# Purpose: This script demonstrates how to generate various statistical summaries for a dataset in Python. It includes basic summary statistics, descriptive statistics with skewness and kurtosis, and a more detailed overview of each variable.
# It uses the Iris dataset as an example.

import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.stats import skew, kurtosis
from sklearn.datasets import load_iris


# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Basic Summary Statistics
print("\nBasic Summary Statistics:")
print(df.describe().to_markdown(numalign="left", stralign="left"))

# Descriptive Statistics with Skewness and Kurtosis
def extended_describe(df):
    """
    Calculates descriptive statistics including skewness and kurtosis.
    """
    stats = df.describe()
    for col in stats.columns:
        if is_numeric_dtype(df[col]):
            stats.loc['skew', col] = skew(df[col].dropna())
            stats.loc['kurtosis', col] = kurtosis(df[col].dropna())
    return stats

print("\nExtended Descriptive Statistics:")
print(extended_describe(df).to_markdown(numalign="left", stralign="left"))


# Detailed Summary for Each Variable
def detailed_describe(df):
    """
    Provides a more detailed summary for each variable.
    """
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            print(f"\nVariable: {col}")
            print(f"  Min: {df[col].min()}, Max: {df[col].max()}")
            print(f"  Mean: {df[col].mean()}, Median: {df[col].median()}")
            print(f"  Std Dev: {df[col].std()}, Skewness: {skew(df[col].dropna())}")
            print(f"  Kurtosis: {kurtosis(df[col].dropna())}")
        else:
            print(f"\nVariable: {col}")
            print(df[col].value_counts().reset_index().rename(columns={'index':'value','count':'frequency'}).to_markdown(index=False,numalign="left", stralign="left"))

print("\nDetailed Summary:")
detailed_describe(df)
