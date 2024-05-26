# Title: Deep Learning Regression in Python with Cross-Validation

# Purpose: This script demonstrates how to build, train, and evaluate a deep learning model for regression using Keras and TensorFlow. 
# It predicts the closing price of Google stock (GOOG) based on other financial indicators (Open, High, Low, Volume, Adjusted).

import pandas as pd
from pandas_datareader import data as pdr
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Fetch Google stock data
symbol = 'GOOG'
start_date = pd.to_datetime('today') - pd.DateOffset(years=10)
end_date = pd.to_datetime('today')
df = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)

# Prepare data for regression
df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adjusted']].copy()
df.dropna(inplace=True)  # Drop any rows with missing values

X = df.drop(columns=['Close'])  # Features (Open, High, Low, Volume, Adjusted)
y = df['Close']  # Target variable (Closing price)

# Scaling Features and Target
scaler_x = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Cross-Validation (K-Folds)
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=7)

fold_metrics = []  # Store metrics for each fold

for fold, (train_index, valid_index) in enumerate(kf.split(X_scaled)):
    print(f"\nFold {fold+1}:")
    X_train, X_valid = X_scaled[train_index], X_scaled[valid_index]
    y_train, y_valid = y_scaled[train_index], y_scaled[valid_index]

    # Build the model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.1),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(1000, activation='relu'), 
        Dense(1, activation='linear')  # Linear activation for regression
    ])

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])  # Mean absolute error as a metric
    model.summary()

    # Early stopping and learning rate reduction
    callbacks = [
        EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
    ]

    # Fit the model
    history = model.fit(X_train, y_train,
                        epochs=10,
                        batch_size=40,
                        validation_split=0.2,
                        callbacks=callbacks,
                        verbose=1)  

    # Evaluate the model
    _, valid_mae = model.evaluate(X_valid, y_valid, verbose=0)
    y_pred = model.predict(X_valid)
    y_pred = scaler_y.inverse_transform(y_pred)  # Reverse scaling for predictions
    y_valid = scaler_y.inverse_transform(y_valid)  # Reverse scaling for true values
    r2 = r2_score(y_valid, y_pred)

    fold_metrics.append({'fold': fold + 1, 'mae': valid_mae, 'r2': r2})


    # Plot training history for each fold
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title(f'Fold {fold + 1} - Training History')
    plt.legend()
    plt.show()

# Aggregate and display metrics across folds
metrics_df = pd.DataFrame(fold_metrics)
print("\nMetrics across folds:")
print(metrics_df.to_markdown(index=False, numalign="left", stralign="left"))

print(f"\nAverage MAE: {metrics_df['mae'].mean():.4f}")
print(f"Average R-squared: {metrics_df['r2'].mean():.4f}")

# Save the model (optional)
# model.save("stock_prediction_model.h5")
