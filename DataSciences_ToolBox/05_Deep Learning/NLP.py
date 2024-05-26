# Title: Text Preprocessing, Annotation, and Filtering using UDPipe

# Purpose: This script performs text preprocessing, part-of-speech (POS) tagging using UDPipe, and filters sentences based on the frequency of specific POS tags.

import pandas as pd
from urllib.request import urlretrieve
import gzip
import os
import re
import spacy_udpipe  # Install using: pip install spacy-udpipe
from collections import Counter

# Download and Load UDPipe Model (only if it hasn't been downloaded before)
model_name = "afrikaans-afribooms"
try:
    udpipe_model = spacy_udpipe.load(model_name)
except OSError:  
    print(f"Downloading UDPipe model: {model_name}")
    spacy_udpipe.download(model_name)
    udpipe_model = spacy_udpipe.load(model_name)


# Load and Preprocess Text Data
data = pd.read_csv("./data/text.csv").tail(100)
data.columns = ["text", "class"]  # Rename columns for clarity

def preprocess_text(text):
    text = text.lower()
    text = text.encode('ascii', 'replace').decode('ascii')  # Remove non-ASCII characters
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    return text

data["text"] = data["text"].astype(str).apply(preprocess_text)


# Annotate Text
def annotate_text(text):
    doc = udpipe_model(text)
    return [(sent.text, token.text, token.pos_) for sent in doc.sents for token in sent]

annotations = data["text"].apply(annotate_text)

# Create DataFrame from Annotations
ann_df = pd.DataFrame(
    [item for sublist in annotations for item in sublist],
    columns=["sentence_id", "token", "upos"]
)

# Reshape and Aggregate Data
ann_df["token_upos"] = ann_df["token"] + "_" + ann_df["upos"]
sent_upos = ann_df.groupby("sentence_id")["token_upos"].apply(list).reset_index()
merged_df = pd.merge(ann_df, sent_upos, on="sentence_id")
merged_df.rename(columns={"token_upos_y": "sent_upos"}, inplace=True)


# Filter Sentences Based on POS Tag Frequency
def count_pos_tags(text):
    pos_tags = [token.split("_")[1] for token in text]
    return Counter(pos_tags)

pos_counts = merged_df["sent_upos"].apply(count_pos_tags)

merged_df["freq_VERB"] = pos_counts.apply(lambda x: x.get("VERB", 0))
merged_df["freq_NOUN"] = pos_counts.apply(lambda x: x.get("NOUN", 0))
merged_df["freq_ADJ"] = pos_counts.apply(lambda x: x.get("ADJ", 0))

fq = 1
filtered_df = merged_df[
    (merged_df["freq_VERB"] > fq) & 
    (merged_df["freq_NOUN"] > fq) & 
    (merged_df["freq_ADJ"] > fq)
]

print("Filtered DataFrame Structure:")
print(filtered_df.info())

# Save the Filtered DataFrame
filtered_df.to_csv("csv/filtered_df.csv", index=False)
