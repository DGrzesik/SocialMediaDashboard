import re

import nltk
import numpy as np
from sentence_transformers import SentenceTransformer

# check if NLTK session is open
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("stopwords")

STOP_WORDS = set(nltk.corpus.stopwords.words('english'))


def clean(text: str) -> str:
    """
    Cleans the input text by removing URLs, user tags, special characters,
    RT string, and converting to lowercase.

    Parameters:
    text (str): The input text to be cleaned.

    Returns:
    str: The cleaned text.
    """
    text = re.sub(r'https?:\/\/[^\s]+', '', text)   # Remove URLs
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)      # Remove user tags
    text = re.sub(r'[^\w\s]', '', text)             # Remove special characters
    text = re.sub('RT', '', text)                   # Remove RT string
    text = re.sub('rsysadmin', '', text)            # Remove rsysadmin string
    text = text.lower()                                         # Convert to lowercase
    return text


def remove_stopwords(text: str) -> str:
    """
    Removes English stopwords from the input text.

    Parameters:
    text (str): The input text from which stopwords will be removed.

    Returns:
    str: The text with stopwords removed.
    """
    words = text.split()
    cleaned_words = [word for word in words if word.lower() not in STOP_WORDS]
    cleaned_text = " ".join(cleaned_words)
    return cleaned_text


def sentence_embedding(clean_text: np.array) -> np.ndarray:
    """
    Creates sentence embeddings using the SentenceTransformer model.

    Parameters:
    clean_text (np.array): The array of cleaned text to be embedded.

    Returns:
    np.ndarray: The array of sentence embeddings.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_embeddings = model.encode(clean_text)
    return np.array(sentence_embeddings)
