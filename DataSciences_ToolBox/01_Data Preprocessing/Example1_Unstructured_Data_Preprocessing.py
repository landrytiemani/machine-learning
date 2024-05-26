# Title: Text Data Preprocessing for Classification

# Purpose: This script demonstrates how to preprocess text data for classification tasks, including cleaning, normalization, and saving the processed data.
# It uses a CSV file as an example input.

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Load Data
data = pd.read_csv("./data/text.csv")
data = data.tail(100)  
data.columns = ['text', 'class']


# Text Preprocessing
def preprocess_text(text):
    """
    Cleans and normalizes a text string.
    """
    text = text.lower()                                # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)                # Remove punctuation
    text = re.sub(r'\d+', '', text)                   # Remove numbers
    text = re.sub(r'br', '', text)                    # Remove "br"
    text = re.sub(r'^na| na ', '', text)              # Remove "na"
    words = word_tokenize(text)                       # Tokenize words
    words = [w for w in words if not w in stopwords.words('english')]  # Remove stopwords
    return ' '.join(words)                            # Join words back into a string


data['text'] = data['text'].astype(str).apply(preprocess_text)

# Merge Preprocessed Text with Labels
merged_data = data[['text', 'class']].copy()
print("\nProcessed Text Data:")
print(merged_data.head().to_markdown(index=False, numalign="left", stralign="left"))

# Save Processed Data
merged_data.to_csv("data/merged_data.csv", index=False)
