# Title: Text Data Preprocessing, DTM Creation, and Preparation for Deep Learning

# Purpose: This script demonstrates how to preprocess text data, create a Document-Term Matrix (DTM) for topic modeling, and prepare the data for deep learning tasks like text classification or text generation.

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load Data
data = pd.read_csv("./data/text.csv")
data = data.tail(100)
data.columns = ['text', 'class']


# Text Preprocessing
def preprocess_text(text):
    """
    Cleans and normalizes a text string.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'br', '', text)
    text = re.sub(r'^na| na ', '', text)
    words = word_tokenize(text)
    words = [w for w in words if not w in stopwords.words('english')]
    return ' '.join(words)

data['text'] = data['text'].astype(str).apply(preprocess_text)


# Document-Term Matrix (DTM) with TF-IDF Weighting
tfidf_vectorizer = TfidfVectorizer(use_idf=True)
dtm = tfidf_vectorizer.fit_transform(data['text'])

# Remove Empty Documents
dtm_array = dtm.toarray()
non_empty_docs = np.where(np.sum(dtm_array, axis=1) > 0)[0]
dtm = dtm[non_empty_docs]

# Merge preprocessed text with labels
merged_data = data[['text', 'class']].copy()


# Data Preparation for Deep Learning
tokenizer = Tokenizer(num_words=4582)  # Adjust num_words based on your vocabulary size
tokenizer.fit_on_texts(data['text'])

text_seqs = tokenizer.texts_to_sequences(data['text'])
padded_seqs = pad_sequences(text_seqs, maxlen=200)  # Adjust maxlen based on your data

# Print outputs
print("\nDocument-Term Matrix (DTM) shape:", dtm.shape)
print(dtm)  # Print the sparse matrix representation of the DTM
print("\nMerged Data (First 5 Rows):")
print(merged_data.head().to_markdown(index=False,numalign="left", stralign="left"))
print("\nPadded Sequences (First 5):\n", padded_seqs[:5])
