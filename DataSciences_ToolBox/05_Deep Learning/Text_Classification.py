# Title: Deep Learning Text Classification in Python (Binary)

# Purpose: This script demonstrates how to preprocess text data, train a deep learning model (CNN+LSTM) for binary text classification, and evaluate its performance using cross-validation.

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, LSTM, Dropout, Bidirectional, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt


# Load Data
data = pd.read_csv("./data/text.csv").tail(100)  # Use the last 100 rows for demonstration
data.columns = ['text', 'class']


# Text Preprocessing
import re
import nltk
from nltk.corpus import stopwords

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

data['text'] = data['text'].astype(str).apply(preprocess_text)


# Balance Classes (optional but recommended for imbalanced datasets)
split_0 = data[data['class'] == 0].sample(frac=1) # Shuffle for randomness
split_1 = data[data['class'] == 1].sample(frac=1)

min_size = min(len(split_0), len(split_1))
data_balanced = pd.concat([split_0[:min_size], split_1[:min_size]])


# Tokenization and Padding
tokenizer = Tokenizer(num_words=10000)  # Set maximum vocabulary size
tokenizer.fit_on_texts(data_balanced['text'])
sequences = tokenizer.texts_to_sequences(data_balanced['text'])
padded_sequences = pad_sequences(sequences, maxlen=2000)


# Prepare Labels
le = LabelEncoder()
y = to_categorical(le.fit_transform(data_balanced['class']))


# Cross-Validation (10-fold)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

fold_metrics = []  # Store metrics for each fold

for fold, (train_index, valid_index) in enumerate(kfold.split(padded_sequences, y.argmax(axis=1))):
    print(f"\nFold {fold+1}:")
    X_train, X_valid = padded_sequences[train_index], padded_sequences[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]

    # Build the Model
    model = Sequential([
        Embedding(10000, 1000, input_length=2000),
        Conv1D(filters=32, kernel_size=5, activation='relu', padding='same'),
        #SpatialDropout1D(0.25),
        MaxPooling1D(pool_size=4),
        Bidirectional(LSTM(32)),
        Dropout(0.5),
        Dense(2, activation='sigmoid')
    ])

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model with Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=10,  
        batch_size=40,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1  
    )
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_valid, y_valid, verbose=0)
    
    # Make predictions
    y_pred = model.predict(X_valid)
    y_pred_classes = np.round(y_pred)
    
    # Classification Report
    report = classification_report(y_valid.argmax(axis=1), y_pred_classes.argmax(axis=1), target_names=["Class 0", "Class 1"])
    print("Classification Report:\n", report)

    fold_metrics.append({'fold': fold + 1, 'loss': loss, 'accuracy': accuracy})
    
    # Plot training and validation loss for each fold
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Fold {fold + 1} - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# Display metrics
metrics_df = pd.DataFrame(fold_metrics)
print("\nMetrics across folds:")
print(metrics_df.to_markdown(index=False, numalign="left", stralign="left"))

print(f"\nAverage Accuracy: {metrics_df['accuracy'].mean():.4f}")
print(f"Average Loss: {metrics_df['loss'].mean():.4f}")


# Save the model (optional)
# model.save("text_classification_model.h5")
