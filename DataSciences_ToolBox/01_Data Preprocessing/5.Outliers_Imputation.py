# Purpose: This script demonstrates how to impute outliers in the Iris dataset using the Local Outlier Factor (LOF) algorithm. Outliers are replaced with the mean of their respective feature.

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import LocalOutlierFactor
import numpy as np


# Load the Iris dataset
data = load_iris()
df = pd.DataFrame(data=data.data, columns=data.feature_names)

# Identify outliers using LOF (Local Outlier Factor)
lof = LocalOutlierFactor(n_neighbors=5, contamination=0.1, novelty=True)  
# novelty=True for prediction purposes

outliers = lof.fit_predict(df)

# Find the indices of outliers
outlier_indices = np.where(outliers == -1)[0]

# Replace outliers with the mean of their respective column
for col in df.columns:
    column_mean = df[col].mean()
    df.loc[outlier_indices, col] = column_mean

# Verify data types (the Iris dataset is already numeric, but included for completeness)
print("\nData types after imputation:")
print(df.dtypes)
