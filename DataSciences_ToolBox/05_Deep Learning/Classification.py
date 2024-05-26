# Title: Deep Learning Classification in Python with Cross-Validation

# Purpose: This script demonstrates how to perform deep learning classification on the Parkinson's Disease dataset using Keras and TensorFlow. It includes data preprocessing, model building, cross-validation, and evaluation.

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler

# Load Data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
df = pd.read_csv(url)
df = df.drop(columns=["name"])  # Remove the name column

# Convert columns to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Separate features and target variable
X = df.drop(columns=["status"])
y = df["status"]

# Scaling Features
scaler = MinMaxScaler()  # Using MinMaxScaler for scaling
X_scaled = scaler.fit_transform(X)

# Create stratified k-fold cross-validator
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

# Cross-Validation
accuracies = []
precisions = []
recalls = []
f1_scores = []

for train, test in kfold.split(X_scaled, y):
    # Prepare training and validation data
    X_train, X_test = X_scaled[train], X_scaled[test]
    y_train, y_test = y[train], y[test]
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    # Build the model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.1),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(1000, activation='relu'), 
        Dense(2, activation='sigmoid')
    ])

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    

    # Fit the model
    history = model.fit(X_train, y_train,
                        epochs=10,
                        batch_size=2,
                        validation_split=0.2,
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
                        verbose=0)  # Set verbose=0 to suppress output
    
    print(f"\nFold Results:")

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.round(y_pred)

    # Classification Report
    report = classification_report(y_test.argmax(axis=1), y_pred_classes.argmax(axis=1), target_names=["healthy", "parkinson's"])
    print("Classification Report:\n", report)
    
    #Append metrics
    accuracies.append(accuracy)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1_score)
    
    # Clear the session to release memory
    tf.keras.backend.clear_session()


# Print average metrics across folds
print("\nAverage Accuracy:", np.mean(accuracies))
print("Average Precision:", np.mean(precisions))
print("Average Recall:", np.mean(recalls))
print("Average F1 Score:", np.mean(f1_scores))

# Save the model (optional)
# model.save("parkinsons_model.h5")
