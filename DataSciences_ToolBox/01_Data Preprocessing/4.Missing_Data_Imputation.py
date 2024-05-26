# Purpose: This script demonstrates how to impute missing values in the Iris dataset using the MICE (Multiple Imputation by Chained Equations) algorithm. It then saves the imputed data to a CSV file.

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.experimental import enable_iterative_imputer  # Enable experimental iterative imputer
from sklearn.impute import IterativeImputer

# Load the Iris dataset
data = load_iris()
df = pd.DataFrame(data=data.data, columns=data.feature_names)

# Introduce some missing values (optional, for demonstration)
# Uncomment these lines to add missing values before imputation
# import numpy as np
# df.iloc[5:10, 1:3] = np.nan  # Add missing values in columns 1 and 2 for rows 5-9

# Impute missing values using MICE (Multiple Imputation by Chained Equations)
imputer = IterativeImputer(max_iter=50, random_state=500)
imputed_data = imputer.fit_transform(df)

# Convert the imputed data back to a DataFrame
imputed_df = pd.DataFrame(imputed_data, columns=df.columns)

# Save the imputed data to a CSV file
imputed_df.to_csv("imputed_iris.csv", index=False)

# Print the structure of the imputed data (optional, for inspection)
print("\nImputed Iris DataFrame:")
print(imputed_df.info())
