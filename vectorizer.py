import spacy
import numpy as np

class SpacyVectorizer:
    def __init__(self, model_name="en_core_web_md"):
        """Initialize the Spacy vectorizer with the specified model."""
        self.nlp = spacy.load(model_name)

    def transform(self, texts):
        """Convert a list of texts into vectors using SpaCy embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        return np.array([self.nlp(text).vector for text in texts], dtype=np.float32)
