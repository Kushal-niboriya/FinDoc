"""
Central project configuration for FinDoc.
Freezes the optimal preprocessing pipeline settings discovered in Phase 2 ablation,
along with model hyperparameters, random seeds, and hardware settings.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class PreprocessingConfig:
    """Frozen optimal preprocessing configuration based on Phase 2 ablation."""
    name: str = "Spec_Full_Dedup"
    lowercase: bool = True
    remove_stopwords: bool = False  # Hurt performance / slow in legal text
    lemmatize: bool = False         # 25x slower with marginal gain
    remove_numbering: bool = True
    mask_money: bool = True
    mask_dates: bool = True
    mask_durations: bool = True
    normalize_parties: bool = True
    collapse_boilerplate: bool = True
    deduplicate: bool = True


@dataclass
class ModelConfig:
    """Standardized model hyperparameters across classical and neural baselines."""
    random_seed: int = 42
    test_size: int = 10000
    val_size: int = 10000
    
    # TF-IDF Features
    tfidf_max_features: int = 25000
    tfidf_ngram_range: tuple = (1, 2)
    tfidf_sublinear_tf: bool = True
    
    # Neural & Transformer Hyperparameters (Tuned for 4GB VRAM)
    transformer_model_name: str = "distilbert-base-uncased"
    sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_seq_length: int = 128
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    learning_rate: float = 3e-5
    num_train_epochs: int = 3
    weight_decay: float = 0.01


# Global default instances
DEFAULT_PREPROCESSING = PreprocessingConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
