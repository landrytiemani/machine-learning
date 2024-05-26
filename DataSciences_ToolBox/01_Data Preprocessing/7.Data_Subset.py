# Purpose: This script demonstrates how to subset (select or remove) columns from a pandas DataFrame. It includes a function for reusable subsetting.

import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Method 1: Direct Removal of Columns
columns_to_remove = ["Petal.Width"]
df_subset = df.drop(columns=columns_to_remove)

print("\nDataFrame after removing Petal.Width:")
print(df_subset.info())


# Method 2: Function for Subsetting Variables
def subset_var(df, col_name):
    """
    Removes a specified column from a DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        col_name (str): The name of the column to remove.

    Returns:
        pandas.DataFrame: The DataFrame with the specified column removed.
    """
    return df.drop(columns=[col_name])

# Example usage of the function
df_subset_again = subset_var(df, "Petal.Width")

print("\nDataFrame after removing Petal.Width using the function:")
print(df_subset_again.info())
