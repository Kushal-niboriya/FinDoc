"""
Feature extraction module for FinDoc.
Builds and manages TF-IDF vectorizers and feature matrices using frozen preprocessing settings.
"""

from typing import Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import DEFAULT_MODEL_CONFIG, DEFAULT_PREPROCESSING
from src.preprocess import PreprocessingPipeline, deduplicate_corpus


def get_default_preprocessor() -> PreprocessingPipeline:
    """Instantiates preprocessor with frozen settings."""
    cfg = DEFAULT_PREPROCESSING
    return PreprocessingPipeline(
        lowercase=cfg.lowercase,
        remove_stopwords=cfg.remove_stopwords,
        lemmatize=cfg.lemmatize,
        remove_numbering=cfg.remove_numbering,
        mask_money=cfg.mask_money,
        mask_dates=cfg.mask_dates,
        mask_durations=cfg.mask_durations,
        normalize_parties=cfg.normalize_parties,
        collapse_boilerplate=cfg.collapse_boilerplate,
    )


def extract_tfidf_features(
    train_texts,
    val_texts,
    test_texts,
    max_features: int = DEFAULT_MODEL_CONFIG.tfidf_max_features,
    ngram_range: tuple = DEFAULT_MODEL_CONFIG.tfidf_ngram_range,
    sublinear_tf: bool = DEFAULT_MODEL_CONFIG.tfidf_sublinear_tf,
) -> Tuple[TfidfVectorizer, Any, Any, Any]:
    """
    Fits TF-IDF on train_texts and transforms train, val, and test texts.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode",
        min_df=2
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    return vectorizer, X_train, X_val, X_test
