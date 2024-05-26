# Title: Structured Data Preprocessing with Outlier Imputation and Normalization

# Purpose: This script demonstrates how to preprocess structured data, focusing on missing value imputation, outlier handling, and normalization.
# It uses the Google stock data from Yahoo Finance as an example.

import pandas as pd
from pandas_datareader import data as pdr
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import numpy as np


# Get Google Stock Data (5-Year Historical)
symbol = 'GOOG'
start_date = pd.to_datetime('today') - pd.DateOffset(years=5)
end_date = pd.to_datetime('today')
data = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)

# Select Relevant Columns and Rename
data = data[['Open', 'High', 'Low', 'Close']]

# Impute Missing Values (if any)
imputer = IterativeImputer(max_iter=50, random_state=500)
imputed_data = imputer.fit_transform(data)

# Convert back to DataFrame and replace missing values with imputed values
data = pd.DataFrame(imputed_data, columns=data.columns)
data = data.dropna()

# Impute Outliers Using LOF
lof = LocalOutlierFactor(n_neighbors=5, contamination=0.1, novelty=True)  
# novelty=True for prediction purposes

outliers = lof.fit_predict(data)

# Find the indices of outliers
outlier_indices = np.where(outliers == -1)[0]


# Replace outliers with the mean of their respective column
for col in data.columns:
    column_mean = data[col].mean()
    data.loc[outlier_indices, col] = column_mean


# Normalize Data (Scale and Center)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Convert back to DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=data.columns)


# Save Processed Data
scaled_df.to_csv("data/processed_GOOG.csv", index=True)  # Save with index (dates)

print("\nProcessed GOOG data saved to data/processed_GOOG.csv")
print(scaled_df.describe().to_markdown())

