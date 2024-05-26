# Title: Deep Learning Text Generation in Python using Harry Potter Data

# Purpose: This script demonstrates how to preprocess text from the Harry Potter books, build and train a CNN-LSTM model for text generation, and generate new text samples.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, LSTM, Dropout, Bidirectional, SpatialDropout1D
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Load and Preprocess Harry Potter Text
# Assuming you have the Harry Potter books as separate .txt files in the 'data/hp_txt' directory
from glob import glob  # For finding files with a pattern
text_files = glob('data/hp_txt/*.txt')

data = ""
for file in text_files:
    with open(file, 'r', encoding='utf-8') as f:
        data += f.read()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    words = word_tokenize(text) 
    words = [w for w in words if w not in stopwords.words('english')]  
    return ' '.join(words)

data = preprocess_text(data)

# Tokenization and Sequence Preparation
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
total_words = len(tokenizer.word_index) + 1  # Add 1 for the out-of-vocabulary token

input_sequences = []
for line in data.split('\n'):  # Split into lines for better context
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Prepare predictors and label
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

# Model Definition
model = Sequential()
model.add(Embedding(total_words, 3000, input_length=max_sequence_len - 1))
model.add(Conv1D(filters=32, kernel_size=5, activation='relu', padding='same'))
model.add(GlobalMaxPooling1D())  # Use global max pooling instead of MaxPooling1D
model.add(LSTM(32))
model.add(Dropout(0.5))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Training
model.fit(X, y, batch_size=40, epochs=10)  # Adjust parameters as needed

# Function to generate text
def generate_text(seed_text, next_words=100, model=model, tokenizer=tokenizer, max_sequence_len=max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word = np.argmax(predicted)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Generate some text (replace with your own seed text)
generated_text = generate_text("Harry Potter was a")
print(generated_text)

# Save the model
model.save('text_generation_model.h5')
