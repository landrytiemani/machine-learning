# Title: Text Data Preprocessing and DTM Creation for Topic Modeling

# Purpose: This script demonstrates how to preprocess text data, including cleaning and normalization, and then create a Document-Term Matrix (DTM) with TF-IDF weighting for topic modeling analysis.

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


# Load Data
data = pd.read_csv("./data/text.csv")
data = data.tail(100)
data.columns = ['text', 'class']


# Text Preprocessing (same as the previous example)
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

# Create Document-Term Matrix (DTM) with TF-IDF Weighting
tfidf_vectorizer = TfidfVectorizer(use_idf=True)
dtm = tfidf_vectorizer.fit_transform(data['text'])

# Remove Empty Documents
dtm_array = dtm.toarray()
non_empty_docs = np.where(np.sum(dtm_array, axis=1) > 0)[0]
dtm = dtm[non_empty_docs]

# Print the DTM
print("\nDocument-Term Matrix (DTM) shape:", dtm.shape)
print(dtm)  # Print the sparse matrix representation of the DTM
