# Title: Structured Data Preprocessing with Outlier Imputation, Balancing, and Normalization (for Classification)

# Purpose: This script demonstrates how to preprocess structured data for classification tasks, including handling categorical variables, imputing missing values, addressing outliers, balancing the dataset, and normalizing features. 
# It uses the Iris dataset as an example.

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
try:
    from NoiseFiltersR import GE
except ImportError:
    !pip install NoiseFiltersR
    from NoiseFiltersR import GE
import numpy as np


# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['Species'] = iris.target_names[iris.target]


# Remove Sepal.Length for demonstration (adjust as needed)
df = df.drop(columns=["Sepal.Length"])


# Impute Missing Values (if any)
imputer = IterativeImputer(max_iter=50, random_state=500)
imputed_data = imputer.fit_transform(df.drop(columns=["Species"]))

# Convert back to DataFrame and replace missing values with imputed values
imputed_df = pd.DataFrame(imputed_data, columns=df.drop(columns=["Species"]).columns)
imputed_df['Species'] = df['Species']
imputed_df = imputed_df.dropna()

# Convert Species column to numeric labels for outlier detection
le = LabelEncoder()
imputed_df["Species_encoded"] = le.fit_transform(imputed_df["Species"])
out = imputed_df.drop(columns=["Species"])

# Impute Outliers Using LOF
lof = LocalOutlierFactor(n_neighbors=5, contamination=0.1, novelty=True)  # novelty=True for prediction purposes

outliers = lof.fit_predict(out)

# Find the indices of outliers
outlier_indices = np.where(outliers == -1)[0]

# Replace outliers with the mean of their respective column
for col in out.columns:
    column_mean = out[col].mean()
    out.loc[outlier_indices, col] = column_mean

# Combine the cleaned data with the target variable
out['Species'] = imputed_df['Species']
df = out



# Balance the Data
# If there's a class imbalance, use undersampling (you can also try oversampling or SMOTE)
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(df.drop(columns=["Species"]), df["Species"])
df = pd.concat([X_resampled, y_resampled], axis=1)


# Normalize the Data (Scale and Center) - Exclude the target variable
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.drop(columns=["Species"]))

# Convert back to DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=df.drop(columns=["Species"]).columns)
df_scaled['Species'] = df['Species']

print("\nProcessed Iris data:")
print(df_scaled.head().to_markdown(index=False,numalign="left", stralign="left"))

# Save Processed Data
df_scaled.to_csv("data/processed_iris.csv", index=False)

