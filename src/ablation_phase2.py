"""
Phase 2: Preprocessing Ablation Study.
Compares 6 preprocessing configurations using TF-IDF + Logistic Regression:
1. Raw Baseline (lowercased)
2. Traditional (Stopword Removal)
3. Traditional (Stopword Removal + Lemmatization)
4. Specialized (Entity Masking: Money, Dates, Durations)
5. Specialized Full (Numbering, Masking, Parties, Boilerplate)
6. Specialized Full + Deduplication

Outputs:
- results/preprocessing_ablation.json
- reports/figures/preprocessing_vocab_reduction.png
- reports/figures/preprocessing_length_comparison.png
- reports/figures/preprocessing_ablation_f1.png
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

from src.data import load_ledgar_offline
from src.preprocess import PreprocessingPipeline, deduplicate_corpus, tokenize_simple

FIGURES_DIR = Path("reports/figures")
RESULTS_DIR = Path("results")
SAMPLE_SIZE = 20000  # Standardized slice for controlled, fast CPU ablation


def run_ablation():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 2: PREPROCESSING ABLATION STUDY (TF-IDF + LOGISTIC REGRESSION)")
    print("=" * 70)

    print("\n[1/4] Loading LEDGAR dataset...")
    ds = load_ledgar_offline()
    raw_train_texts = ds["train"]["text"][:SAMPLE_SIZE]
    raw_train_labels = ds["train"]["label"][:SAMPLE_SIZE]
    raw_val_texts = ds["validation"]["text"]
    val_labels = ds["validation"]["label"]

    print(f"Train subset: {len(raw_train_texts):,} examples | Validation set: {len(raw_val_texts):,} examples")

    # Define the 6 experimental configurations
    configs = {
        "1. Raw Baseline": PreprocessingPipeline(
            lowercase=True
        ),
        "2. Trad: Stopwords": PreprocessingPipeline(
            lowercase=True,
            remove_stopwords=True
        ),
        "3. Trad: Stopwords + Lemmatization": PreprocessingPipeline(
            lowercase=True,
            remove_stopwords=True,
            lemmatize=True
        ),
        "4. Spec: Masking Only": PreprocessingPipeline(
            lowercase=True,
            mask_money=True,
            mask_dates=True,
            mask_durations=True
        ),
        "5. Spec: Full Legal": PreprocessingPipeline(
            lowercase=True,
            remove_numbering=True,
            mask_money=True,
            mask_dates=True,
            mask_durations=True,
            normalize_parties=True,
            collapse_boilerplate=True
        ),
        "6. Spec: Full + Dedup": PreprocessingPipeline(
            lowercase=True,
            remove_numbering=True,
            mask_money=True,
            mask_dates=True,
            mask_durations=True,
            normalize_parties=True,
            collapse_boilerplate=True
        ),
    }

    results = []
    processed_val_cache = {}
    transformed_sample_lengths = {}

    print("\n[2/4] Running ablation across 6 configurations...")
    for cfg_idx, (name, pipe) in enumerate(configs.items(), start=1):
        print(f"\n---> Running Config {name}...")
        t_start = time.perf_counter()

        # Transform training data
        t_prep_start = time.perf_counter()
        # For lemmatization, run in chunks to monitor progress
        print(f"  Transforming {len(raw_train_texts)} train texts...")
        train_texts = pipe.transform_batch(raw_train_texts, batch_size=2000)
        train_labels = list(raw_train_labels)

        # Apply deduplication if config 6
        num_dedup_removed = 0
        if "Dedup" in name:
            train_texts, train_labels, num_dedup_removed = deduplicate_corpus(train_texts, train_labels)
            print(f"  Deduplication removed {num_dedup_removed} duplicate training samples.")

        # Transform validation data (cache if identical pipe)
        pipe_key = str(vars(pipe))
        if pipe_key not in processed_val_cache:
            print(f"  Transforming {len(raw_val_texts)} validation texts...")
            processed_val_cache[pipe_key] = pipe.transform_batch(raw_val_texts, batch_size=2000)
        val_texts = processed_val_cache[pipe_key]

        prep_time = time.perf_counter() - t_prep_start

        # Measure vocabulary and token lengths
        all_tokens = [t for text in train_texts for t in tokenize_simple(text)]
        vocab_size = len(set(all_tokens))
        sample_lengths = [len(tokenize_simple(text)) for text in train_texts]
        mean_len = float(np.mean(sample_lengths))
        median_len = float(np.median(sample_lengths))
        transformed_sample_lengths[name] = sample_lengths

        # Fit TF-IDF Vectorizer
        t_fit_start = time.perf_counter()
        vec = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)
        X_train = vec.fit_transform(train_texts)
        X_val = vec.transform(val_texts)

        # Fit Logistic Regression
        clf = LogisticRegression(max_iter=150, C=1.0, random_state=42, solver="lbfgs")
        clf.fit(X_train, train_labels)
        fit_time = time.perf_counter() - t_fit_start

        # Evaluate
        preds = clf.predict(X_val)
        macro_f1 = f1_score(val_labels, preds, average="macro", zero_division=0)
        micro_f1 = f1_score(val_labels, preds, average="micro", zero_division=0)
        acc = accuracy_score(val_labels, preds)

        total_time = time.perf_counter() - t_start

        cfg_summary = {
            "config_name": name,
            "vocab_size": vocab_size,
            "mean_token_length": round(mean_len, 2),
            "median_token_length": round(median_len, 2),
            "macro_f1": round(float(macro_f1), 4),
            "micro_f1": round(float(micro_f1), 4),
            "accuracy": round(float(acc), 4),
            "prep_time_sec": round(prep_time, 2),
            "fit_time_sec": round(fit_time, 2),
            "total_time_sec": round(total_time, 2),
            "train_samples_used": len(train_texts),
            "dedup_removed": num_dedup_removed,
        }
        results.append(cfg_summary)
        print(f"  Result -> Macro-F1: {macro_f1:.4f} | Micro-F1: {micro_f1:.4f} | Acc: {acc:.4f} | Vocab: {vocab_size:,} | Total Time: {total_time:.1f}s")

    # Find the best configuration
    best_config = max(results, key=lambda x: x["macro_f1"])
    print("\n" + "=" * 70)
    print(f"BEST CONFIGURATION: {best_config['config_name']} (Macro-F1: {best_config['macro_f1']})")
    print("=" * 70)

    # Save to results/preprocessing_ablation.json
    ablation_payload = {
        "metadata": {
            "model": "TF-IDF (15k max_features, ngram 1-2, sublinear_tf) + LogisticRegression(C=1.0, max_iter=150)",
            "train_sample_size": SAMPLE_SIZE,
            "val_sample_size": len(raw_val_texts),
            "random_seed": 42
        },
        "configurations": results,
        "best_configuration": best_config,
    }
    with open(RESULTS_DIR / "preprocessing_ablation.json", "w", encoding="utf-8") as f:
        json.dump(ablation_payload, f, indent=2)

    # [3/4] Generate Visualizations
    print("\n[3/4] Generating comparative plots...")

    # Plot 1: Vocabulary Size Reduction
    plt.figure(figsize=(10, 5))
    names = [r["config_name"] for r in results]
    vocabs = [r["vocab_size"] for r in results]
    bars = plt.barh(names[::-1], vocabs[::-1], color="#1f77b4", alpha=0.85)
    plt.title("Vocabulary Size Comparison Across Preprocessing Pipelines", fontsize=12, fontweight="bold")
    plt.xlabel("Unique Vocabulary Size (Training Set)", fontsize=10)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 200, bar.get_y() + bar.get_height() / 2, f"{int(w):,}", va="center", fontsize=9)
    plt.xlim(0, max(vocabs) * 1.15)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "preprocessing_vocab_reduction.png", dpi=300)
    plt.close()

    # Plot 2: Length Distribution Comparison (Density plot for 3 key variants)
    plt.figure(figsize=(10, 5))
    bins = np.linspace(0, 250, 50)
    for col_cfg, col_color in [
        ("1. Raw Baseline", "#1f77b4"),
        ("2. Trad: Stopwords", "#ff7f0e"),
        ("5. Spec: Full Legal", "#2ca02c")
    ]:
        lens = transformed_sample_lengths[col_cfg]
        plt.hist(lens, bins=bins, density=True, alpha=0.45, label=f"{col_cfg} (Mean={np.mean(lens):.1f})", color=col_color)
    plt.axvline(128, color="red", linestyle="--", label="128-token boundary")
    plt.title("Clause Length Compression Under Different Preprocessing Pipelines", fontsize=12, fontweight="bold")
    plt.xlabel("Tokens per Clause", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "preprocessing_length_comparison.png", dpi=300)
    plt.close()

    # Plot 3: Macro-F1 Ablation Comparison
    plt.figure(figsize=(10, 5))
    macro_f1s = [r["macro_f1"] for r in results]
    accs = [r["accuracy"] for r in results]
    y_pos = np.arange(len(names))
    
    plt.barh(y_pos - 0.15, macro_f1s[::-1], height=0.3, label="Macro-F1 (Headline)", color="#2ca02c", alpha=0.9)
    plt.barh(y_pos + 0.15, accs[::-1], height=0.3, label="Accuracy", color="#1f77b4", alpha=0.6)
    
    plt.yticks(y_pos, names[::-1], fontsize=9)
    plt.xlabel("Score on Validation Split", fontsize=10)
    plt.title("Impact of Preprocessing Pipeline on Classifier Performance", fontsize=12, fontweight="bold")
    plt.xlim(0.5, 0.9)
    plt.legend(loc="lower right")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    
    for i, (f1, ac) in enumerate(zip(macro_f1s[::-1], accs[::-1])):
        plt.text(f1 + 0.005, y_pos[i] - 0.15, f"{f1:.4f}", va="center", fontsize=8, fontweight="bold", color="darkgreen")
        plt.text(ac + 0.005, y_pos[i] + 0.15, f"{ac:.4f}", va="center", fontsize=8, color="navy")
        
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "preprocessing_ablation_f1.png", dpi=300)
    plt.close()

    print("\n[4/4] Completed! Saved ablation results to results/preprocessing_ablation.json and plots to reports/figures/")
    return results, best_config


if __name__ == "__main__":
    run_ablation()
