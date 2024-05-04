# data/preprocessing.py
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_text(text):
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the text data
    vectorized_text = vectorizer.fit_transform(text)

    return vectorized_text
