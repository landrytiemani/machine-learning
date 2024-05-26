# Title: Bar Plots with Points, Lines, and Error Bars (and Time Series Plot) in Python

# Purpose: This script demonstrates how to create various plots:
# 1. Bar plot with points, lines, and error bars to visualize mean and confidence intervals.
# 2. Time series plot to visualize the closing price of Google stock.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from pandas_datareader import data as pdr

# Section 1: Bar Plot with Points, Lines, and Error Bars

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['Species'] = iris.target_names[iris.target]

# Create a bar plot with points, lines, and error bars
plt.figure(figsize=(10, 6))

# Bar plot with error bars (95% confidence interval by default)
sns.barplot(x='Species', y='Sepal.Length', data=df, color="white", errcolor=".2", edgecolor=".2")

# Add points to the bar plot
sns.pointplot(x='Species', y='Sepal.Length', data=df, color="black", join=False)

# Add a line connecting the mean points
sns.lineplot(x='Species', y='Sepal.Length', data=df, color="blue", estimator='mean', ci=None)


# Customize the plot appearance
sns.despine(left=True)  # Remove the left spine for a cleaner look
plt.xlabel('Species', fontsize=12)
plt.ylabel('Sepal Length', fontsize=12)
plt.title('Mean Sepal Length by Species with Confidence Intervals', fontsize=14)

plt.show()

# Section 2: Time Series Plot (Google Stock Closing Price)

# Fetch Google stock data (replace 'GOOG' with your desired ticker symbol)
symbol = 'GOOG'
start_date = pd.to_datetime('today') - pd.DateOffset(years=5)
end_date = pd.to_datetime('today')
df_stock = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)

# Plot the closing price
plt.figure(figsize=(12, 6))
plt.plot(df_stock.index, df_stock['Close'])

# Customize the plot appearance
plt.title(f'Close Price of {symbol}', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close Price', fontsize=12)
plt.grid(axis='y', alpha=0.5)

plt.show()
