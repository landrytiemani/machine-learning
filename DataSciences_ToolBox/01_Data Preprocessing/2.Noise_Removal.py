# Purpose: This code automatically filters noise from the Iris dataset using the GE (Generalized Editing) algorithm from the NoiseFiltersR library. 
# It then displays the structure of the cleaned data and the indices of rows removed as noise.

import pandas as pd
import pandas as pd
from sklearn.datasets import load_iris

# Attempt to import the NoiseFiltersR library, if not found, install it
try:
    from NoiseFiltersR import GE
except ImportError:
    !pip install NoiseFiltersR  # Install the NoiseFiltersR library
    from NoiseFiltersR import GE

# Load the Iris dataset
data = load_iris()

# Create a pandas DataFrame from the Iris data
df = pd.DataFrame(data=data.data, columns=data.feature_names)

# Convert the 'Species' column to a categorical data type
df['Species'] = pd.Categorical(data.target)

# Prepare the formula (Species ~ .) for the noise filter
target = 'Species'
formula = f"{target} ~ ."

# Apply the GE noise filter (k=5, kk=ceiling(5/2))
noise_filter = GE(formula, data=df, k=5, kk=5//2+1)

# Retrieve the cleaned data after applying the noise filter
clean_data = noise_filter.cleanData

# Print the structure of the cleaned data
print(clean_data.info())

# Print the row indices removed by the noise filter
print(noise_filter.repIdx)