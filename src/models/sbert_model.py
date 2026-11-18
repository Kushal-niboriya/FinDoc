"""
Frozen Sentence Embeddings Model for FinDoc.
Extracts 384-dimensional dense semantic embeddings using:
sentence-transformers/all-MiniLM-L6-v2 on GPU with fp16.
Trains a Multinomial Logistic Regression classifier on top of the frozen embeddings.
Evaluates on validation and test sets.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

from src.data import load_ledgar_offline, get_dataset_metadata
from src.features import get_default_preprocessor
from src.preprocess import deduplicate_corpus

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 128


def run_sbert_experiment():
    print("=" * 70)
    print(f"FROZEN SENTENCE EMBEDDINGS: {MODEL_NAME} + LOGISTIC REGRESSION")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")

    ds = load_ledgar_offline()
    meta = get_dataset_metadata(ds)
    label_names = meta["label_names"]
    num_classes = len(label_names)

    train_counts = np.bincount(ds["train"]["label"], minlength=num_classes)
    sorted_classes = np.argsort(-train_counts)
    class_tiers = {
        "head": sorted_classes[:15].tolist(),
        "mid": sorted_classes[15:50].tolist(),
        "tail": sorted_classes[50:].tolist(),
    }

    print("\n[1/4] Applying frozen preprocessing pipeline...")
    pipe = get_default_preprocessor()
    train_texts = pipe.transform_batch(ds["train"]["text"])
    val_texts = pipe.transform_batch(ds["validation"]["text"])
    test_texts = pipe.transform_batch(ds["test"]["text"])

    train_texts, train_labels, _ = deduplicate_corpus(train_texts, ds["train"]["label"])
    val_labels = ds["validation"]["label"]
    test_labels = ds["test"]["label"]

    print("\n[2/4] Loading SentenceTransformer encoder...")
    encoder = SentenceTransformer(MODEL_NAME, device=device)
    if device == "cuda":
        encoder.half() # fp16 for speed and low memory
        torch.cuda.reset_peak_memory_stats()

    t_enc_start = time.perf_counter()
    print(f"Encoding {len(train_texts):,} train clauses (batch_size={BATCH_SIZE})...")
    X_train = encoder.encode(train_texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)
    print(f"Encoding {len(val_texts):,} validation clauses...")
    X_val = encoder.encode(val_texts, batch_size=BATCH_SIZE, show_progress_bar=False, normalize_embeddings=True)
    
    t_test_enc_start = time.perf_counter()
    print(f"Encoding {len(test_texts):,} test clauses...")
    X_test = encoder.encode(test_texts, batch_size=BATCH_SIZE, show_progress_bar=False, normalize_embeddings=True)
    test_enc_time = time.perf_counter() - t_test_enc_start
    total_enc_time = time.perf_counter() - t_enc_start

    latency_ms_per_sample = (test_enc_time / len(test_texts)) * 1000

    peak_vram_mb = 0.0
    if device == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    print(f"Encoding completed in {total_enc_time:.2f}s | Shape: {X_train.shape} | Peak VRAM: {peak_vram_mb:.1f} MB")

    print("\n[3/4] Fitting Logistic Regression on 384-d dense embeddings...")
    t_fit_start = time.perf_counter()
    clf = LogisticRegression(C=2.0, max_iter=250, random_state=42, solver="lbfgs")
    clf.fit(X_train, train_labels)
    fit_time = time.perf_counter() - t_fit_start
    print(f"Classifier fit completed in {fit_time:.2f}s")

    print("\n[4/4] Evaluating on Validation and Test sets...")
    labels_all = list(range(num_classes))
    val_preds = clf.predict(X_val)
    val_macro = f1_score(val_labels, val_preds, labels=labels_all, average="macro", zero_division=0)
    val_acc = accuracy_score(val_labels, val_preds)

    test_preds = clf.predict(X_test)
    test_macro = f1_score(test_labels, test_preds, labels=labels_all, average="macro", zero_division=0)
    test_micro = f1_score(test_labels, test_preds, labels=labels_all, average="micro", zero_division=0)
    test_acc = accuracy_score(test_labels, test_preds)
    per_class_f1 = f1_score(test_labels, test_preds, labels=labels_all, average=None, zero_division=0)

    head_f1 = float(np.mean([per_class_f1[i] for i in class_tiers["head"]]))
    mid_f1 = float(np.mean([per_class_f1[i] for i in class_tiers["mid"]]))
    tail_f1 = float(np.mean([per_class_f1[i] for i in class_tiers["tail"]]))

    print(f"\n--> Frozen SBERT Results:")
    print(f"    Val Macro-F1:  {val_macro:.4f} | Val Accuracy: {val_acc:.4f}")
    print(f"    Test Macro-F1: {test_macro:.4f} | Test Micro-F1: {test_micro:.4f} | Test Accuracy: {test_acc:.4f}")
    print(f"    Tier Breakdown: Head={head_f1:.4f} | Mid={mid_f1:.4f} | Tail={tail_f1:.4f}")
    print(f"    Inference Latency: {latency_ms_per_sample:.3f} ms/sample")
    print(f"    Peak VRAM: {peak_vram_mb:.1f} MB")

    param_count = sum(p.numel() for p in encoder.parameters())

    return {
        "model_name": "Frozen SBERT (all-MiniLM-L6-v2) + LogisticRegression",
        "parameters": param_count,
        "embedding_dim": 384,
        "train_time_sec": round(total_enc_time + fit_time, 2),
        "encoding_time_sec": round(total_enc_time, 2),
        "fit_time_sec": round(fit_time, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "latency_ms_per_sample": round(latency_ms_per_sample, 3),
        "val_macro_f1": round(float(val_macro), 4),
        "test_metrics": {
            "macro_f1": round(float(test_macro), 4),
            "micro_f1": round(float(test_micro), 4),
            "accuracy": round(float(test_acc), 4),
            "tier_f1": {
                "head_f1": round(head_f1, 4),
                "mid_f1": round(mid_f1, 4),
                "tail_f1": round(tail_f1, 4),
            }
        }
    }


if __name__ == "__main__":
    run_sbert_experiment()
