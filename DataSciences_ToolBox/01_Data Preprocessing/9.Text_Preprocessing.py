# Purpose: This script demonstrates text preprocessing (cleaning, transformation) and tokenization for text data loaded from a CSV file.

import pandas as pd
import re
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer


# Load data from the CSV file
data = pd.read_csv("./data/text.csv")
data = data.tail(100)
data.columns = ['text', 'class']


# Text Preprocessing
def preprocess_text(text):
    text = text.lower()                                # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)                # Remove punctuation
    text = re.sub(r'\d+', '', text)                   # Remove numbers
    text = re.sub(r'br', '', text)                    # Remove "br"
    text = re.sub(r'^na| na ', '', text)              # Remove "na"
    words = word_tokenize(text)                       # Tokenize words
    words = [w for w in words if not w in stopwords.words('english')]  # Remove stopwords
    return ' '.join(words)                            # Join words back into a string

data['text'] = data['text'].astype(str).apply(preprocess_text)  # Apply preprocessing


# Merge text with labels
merged_data = data[['text', 'class']].copy()
merged_data.to_csv("merged_data.csv", index=False)  # Save to CSV


# Tokenization with Keras
tokenizer = Tokenizer(num_words=4582)
tokenizer.fit_on_texts(data['text'])

text_seqs = tokenizer.texts_to_sequences(data['text'])

padded_seqs = pad_sequences(text_seqs, maxlen=200)


# Optional: Document-Term Matrix (DTM) with TF-IDF using scikit-learn
tfidf_vectorizer = TfidfVectorizer(use_idf=True)
dtm = tfidf_vectorizer.fit_transform(data['text'])  # Create DTM

# Display outputs (first few rows for demonstration)
print("\nPreprocessed Data:")
print(merged_data.head().to_markdown(index=False, numalign="left", stralign="left"))  # Use markdown format for better display
print("\nTokenized Sequences (first example):", text_seqs[0])
print("\nPadded Sequences (first few):")
print(padded_seqs[:5])
print("\nDocument-Term Matrix (DTM) shape:", dtm.shape)
