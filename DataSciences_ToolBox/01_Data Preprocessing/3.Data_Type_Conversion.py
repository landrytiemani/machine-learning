# Purpose: This script demonstrates various data type conversions in Python, including string to numeric, numeric to factor (categorical), numeric to string, and string to date.

import pandas as pd


# Numerical Conversion
x = "1"
print(f"Original type: {type(x)}")  # Output: Original type: <class 'str'>

x = float(x)  # Or int(x) for integer conversion
print(f"Converted type: {type(x)}")  # Output: Converted type: <class 'float'>


# Factor Conversion
x = 1
print(f"Original type: {type(x)}")  # Output: Original type: <class 'int'>

x = pd.Categorical([x])  # Use pandas Categorical for factor-like behavior
print(f"Converted type: {type(x)}")  # Output: Converted type: <class 'pandas.core.arrays.categorical.Categorical'>
print(f"Converted values: {x}")     # Output: Converted values: [1, NaN, NaN, NaN, NaN, ...]  (default categories)


# Character Conversion
x = 1
print(f"Original type: {type(x)}")  # Output: Original type: <class 'int'>

x = str(x)
print(f"Converted type: {type(x)}")  # Output: Converted type: <class 'str'>


# Date Conversion
x = "01-11-2023"
print(f"Original type: {type(x)}")  # Output: Original type: <class 'str'>

x = pd.to_datetime(x, format="%m-%d-%Y")
print(f"Converted type: {type(x)}")  # Output: Converted type: <class 'pandas._libs.tslibs.timestamps.Timestamp'>
print(f"Converted value: {x}")      # Output: Converted value: 2023-01-11 00:00:00
