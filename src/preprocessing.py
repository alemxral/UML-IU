"""
Text preprocessing module for scientific paper analysis.
Handles cleaning, tokenization, and feature extraction.
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Download required NLTK data (with error handling for space issues)
try:
    nltk.data.find('tokenizers/punkt_tab')
except (LookupError, OSError, Exception):
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        pass  # Skip if download fails
    
try:
    nltk.data.find('corpora/stopwords')
except (LookupError, OSError, Exception):
    try:
        nltk.download('stopwords', quiet=True)
    except:
        pass  # Skip if download fails
    
try:
    nltk.data.find('corpora/wordnet')
except (LookupError, OSError, Exception):
    try:
        nltk.download('wordnet', quiet=True)
    except:
        pass  # Skip if download fails


class TextPreprocessor:
    """Comprehensive text preprocessing for scientific abstracts."""
    
    def __init__(self, use_lemmatization=True):
        """
        Initialize the preprocessor.
        
        Args:
            use_lemmatization: Whether to apply lemmatization
        """
        self.use_lemmatization = use_lemmatization
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        
        # Scientific stopwords - common terms in abstracts
        self.stop_words = set(stopwords.words('english'))
        self.scientific_stopwords = {
            'paper', 'study', 'research', 'method', 'approach', 'result', 
            'show', 'present', 'propose', 'also', 'using', 'used', 'use',
            'based', 'model', 'data', 'work', 'new', 'propose', 'problem'
        }
        self.stop_words.update(self.scientific_stopwords)
        
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters, numbers, and extra whitespace.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove LaTeX commands and equations
        text = re.sub(r'\$.*?\$', '', text)  # Inline equations
        text = re.sub(r'\\[a-z]+\{.*?\}', '', text)  # LaTeX commands
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_filter(self, text: str) -> List[str]:
        """
        Tokenize text and filter out stopwords and short tokens.
        
        Args:
            text: Cleaned text string
            
        Returns:
            List of filtered tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter stopwords and short tokens
        tokens = [
            token for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        # Apply lemmatization if enabled
        if self.use_lemmatization and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def preprocess(self, text: str) -> str:
        """
        Complete preprocessing pipeline: clean + tokenize + rejoin.
        
        Args:
            text: Raw text string
            
        Returns:
            Preprocessed text string
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_filter(cleaned)
        return ' '.join(tokens)
    
    def preprocess_corpus(self, texts: pd.Series, show_progress=True) -> pd.Series:
        """
        Preprocess an entire corpus of texts.
        
        Args:
            texts: Series of text strings
            show_progress: Whether to show progress bar
            
        Returns:
            Series of preprocessed texts
        """
        if show_progress:
            from tqdm import tqdm
            tqdm.pandas(desc="Preprocessing texts")
            return texts.progress_apply(self.preprocess)
        else:
            return texts.apply(self.preprocess)


class FeatureExtractor:
    """Extract features from preprocessed text using TF-IDF."""
    
    def __init__(self, max_features=5000, ngram_range=(1, 2), min_df=5, max_df=0.8):
        """
        Initialize the feature extractor.
        
        Args:
            max_features: Maximum number of features to extract
            ngram_range: Range of n-grams to consider
            min_df: Minimum document frequency
            max_df: Maximum document frequency (as proportion)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True
        )
        
    def fit_transform(self, texts: pd.Series) -> Tuple[np.ndarray, List[str]]:
        """
        Fit the vectorizer and transform texts to TF-IDF features.
        
        Args:
            texts: Series of preprocessed texts
            
        Returns:
            Tuple of (feature matrix, feature names)
        """
        X = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()
        return X.toarray(), list(feature_names)
    
    def transform(self, texts: pd.Series) -> np.ndarray:
        """
        Transform texts using fitted vectorizer.
        
        Args:
            texts: Series of preprocessed texts
            
        Returns:
            Feature matrix
        """
        return self.vectorizer.transform(texts).toarray()
    
    def get_top_terms_per_document(self, X: np.ndarray, feature_names: List[str], 
                                   top_n=10) -> List[List[Tuple[str, float]]]:
        """
        Get top N terms for each document.
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            top_n: Number of top terms to extract
            
        Returns:
            List of lists containing (term, score) tuples
        """
        top_terms = []
        for doc_vec in X:
            top_indices = doc_vec.argsort()[-top_n:][::-1]
            doc_top_terms = [
                (feature_names[idx], doc_vec[idx]) 
                for idx in top_indices if doc_vec[idx] > 0
            ]
            top_terms.append(doc_top_terms)
        return top_terms


def extract_top_keywords(X: np.ndarray, feature_names: List[str], 
                        cluster_labels: np.ndarray, top_n=20) -> dict:
    """
    Extract top keywords for each cluster based on TF-IDF scores.
    
    Args:
        X: TF-IDF feature matrix
        feature_names: List of feature names
        cluster_labels: Cluster assignments for each document
        top_n: Number of top keywords to extract per cluster
        
    Returns:
        Dictionary mapping cluster ID to list of (keyword, score) tuples
    """
    cluster_keywords = {}
    
    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:  # Skip noise points in DBSCAN
            continue
            
        # Get documents in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_tfidf = X[cluster_mask]
        
        # Calculate mean TF-IDF score for each term in the cluster
        mean_tfidf = np.asarray(cluster_tfidf.mean(axis=0)).flatten()
        
        # Get top terms
        top_indices = mean_tfidf.argsort()[-top_n:][::-1]
        keywords = [
            (feature_names[idx.item() if hasattr(idx, 'item') else int(idx)], 
             mean_tfidf[idx].item() if hasattr(mean_tfidf[idx], 'item') else float(mean_tfidf[idx]))
            for idx in top_indices
        ]
        
        cluster_keywords[cluster_id] = keywords
    
    return cluster_keywords
