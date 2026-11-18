"""
Classical ML Baselines for FinDoc.
Models implemented:
1. Majority Class (DummyClassifier)
2. TF-IDF + Logistic Regression
3. TF-IDF + Linear SVM (LinearSVC)
4. TF-IDF + Complement Naive Bayes (ComplementNB)

Hyperparameter tuning with documented grid on validation set.
Full evaluation on Validation and Test sets with Head/Mid/Tail analysis.
Outputs:
- results/baselines_results.json
- reports/figures/baselines_performance_comparison.png
- reports/figures/baselines_head_tail_f1.png
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import f1_score, accuracy_score, classification_report

from src.data import load_ledgar_offline, get_dataset_metadata
from src.features import get_default_preprocessor, extract_tfidf_features
from src.preprocess import deduplicate_corpus

FIGURES_DIR = Path("reports/figures")
RESULTS_DIR = Path("results")


def evaluate_predictions(y_true, y_pred, label_names, class_tiers):
    """
    Computes macro-F1, micro-F1, accuracy, per-class F1, and head/mid/tail breakdowns.
    """
    num_classes = len(label_names)
    labels_all = list(range(num_classes))
    macro_f1 = f1_score(y_true, y_pred, labels=labels_all, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, labels=labels_all, average="micro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    per_class_f1 = f1_score(y_true, y_pred, labels=labels_all, average=None, zero_division=0)
    per_class_dict = {
        label_names[i]: round(float(per_class_f1[i]), 4)
        for i in range(num_classes)
    }

    # Head, Mid, Tail breakdown
    head_indices = class_tiers["head"]
    mid_indices = class_tiers["mid"]
    tail_indices = class_tiers["tail"]

    head_f1 = float(np.mean([per_class_f1[i] for i in head_indices]))
    mid_f1 = float(np.mean([per_class_f1[i] for i in mid_indices]))
    tail_f1 = float(np.mean([per_class_f1[i] for i in tail_indices]))

    return {
        "macro_f1": round(float(macro_f1), 4),
        "micro_f1": round(float(micro_f1), 4),
        "accuracy": round(float(acc), 4),
        "tier_f1": {
            "head_f1 (ranks 1-15)": round(head_f1, 4),
            "mid_f1 (ranks 16-50)": round(mid_f1, 4),
            "tail_f1 (ranks 51-100)": round(tail_f1, 4),
        },
        "per_class_f1": per_class_dict
    }


def run_baselines():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("PHASE 3: CLASSICAL BASELINE SUITE & HEAD-VS-TAIL ANALYSIS")
    print("=" * 75)

    print("\n[1/5] Loading LEDGAR dataset and metadata...")
    ds = load_ledgar_offline()
    meta = get_dataset_metadata(ds)
    label_names = meta["label_names"]
    num_classes = len(label_names)

    raw_train_texts = ds["train"]["text"]
    raw_train_labels = ds["train"]["label"]
    raw_val_texts = ds["validation"]["text"]
    raw_val_labels = ds["validation"]["label"]
    raw_test_texts = ds["test"]["text"]
    raw_test_labels = ds["test"]["label"]

    print(f"Loaded: Train={len(raw_train_texts):,}, Val={len(raw_val_texts):,}, Test={len(raw_test_texts):,}")

    # Determine class tiers based on training frequency
    train_counts = np.bincount(raw_train_labels, minlength=num_classes)
    sorted_class_indices = np.argsort(-train_counts)  # rank 1 to 100
    class_tiers = {
        "head": sorted_class_indices[:15].tolist(),
        "mid": sorted_class_indices[15:50].tolist(),
        "tail": sorted_class_indices[50:].tolist(),
    }

    print("\n[2/5] Applying frozen preprocessing pipeline...")
    pipe = get_default_preprocessor()
    t0 = time.perf_counter()
    train_texts = pipe.transform_batch(raw_train_texts)
    val_texts = pipe.transform_batch(raw_val_texts)
    test_texts = pipe.transform_batch(raw_test_texts)
    print(f"Preprocessing completed in {time.perf_counter() - t0:.2f}s")

    # Deduplicate training split
    train_texts, train_labels, removed_count = deduplicate_corpus(train_texts, raw_train_labels)
    print(f"Deduplication removed {removed_count:,} duplicate clauses. Training set size: {len(train_texts):,}")

    # Extract TF-IDF features
    print("\n[3/5] Extracting TF-IDF feature matrices (max_features=25000, n-gram=(1,2), sublinear_tf=True)...")
    t0 = time.perf_counter()
    vectorizer, X_train, X_val, X_test = extract_tfidf_features(
        train_texts, val_texts, test_texts,
        max_features=25000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    print(f"TF-IDF matrices built in {time.perf_counter() - t0:.2f}s | Shape: {X_train.shape}")

    models_results = {}
    tuning_log = {}

    # 1. Majority Class Dummy Baseline
    print("\n---> Model 1: Majority Class Baseline...")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train_texts, train_labels)
    dummy_val_preds = dummy.predict(val_texts)
    dummy_test_preds = dummy.predict(test_texts)
    
    models_results["Majority Class"] = {
        "best_params": {"strategy": "most_frequent"},
        "val_metrics": evaluate_predictions(raw_val_labels, dummy_val_preds, label_names, class_tiers),
        "test_metrics": evaluate_predictions(raw_test_labels, dummy_test_preds, label_names, class_tiers),
        "fit_time_sec": 0.01,
    }
    print(f"  Majority Class Test Macro-F1: {models_results['Majority Class']['test_metrics']['macro_f1']:.4f}")

    # 2. Complement Naive Bayes (Hyperparameter Tuning: alpha in [0.1, 0.5, 1.0])
    print("\n---> Model 2: Complement Naive Bayes (Tuning alpha in [0.1, 0.5, 1.0])...")
    cnb_grid = [0.1, 0.5, 1.0]
    best_cnb_score = -1.0
    best_cnb_model = None
    best_alpha = None
    cnb_tuning = {}

    for alpha in cnb_grid:
        clf = ComplementNB(alpha=alpha, norm=True)
        clf.fit(X_train, train_labels)
        preds = clf.predict(X_val)
        score = f1_score(raw_val_labels, preds, average="macro", zero_division=0)
        cnb_tuning[f"alpha_{alpha}"] = round(float(score), 4)
        print(f"  CNB alpha={alpha} -> Val Macro-F1: {score:.4f}")
        if score > best_cnb_score:
            best_cnb_score = score
            best_cnb_model = clf
            best_alpha = alpha

    tuning_log["ComplementNB"] = cnb_tuning
    t_start = time.perf_counter()
    best_cnb_model.fit(X_train, train_labels)
    cnb_fit_time = time.perf_counter() - t_start
    cnb_val_preds = best_cnb_model.predict(X_val)
    cnb_test_preds = best_cnb_model.predict(X_test)

    models_results["Complement Naive Bayes"] = {
        "best_params": {"alpha": best_alpha, "norm": True},
        "val_metrics": evaluate_predictions(raw_val_labels, cnb_val_preds, label_names, class_tiers),
        "test_metrics": evaluate_predictions(raw_test_labels, cnb_test_preds, label_names, class_tiers),
        "fit_time_sec": round(cnb_fit_time, 2),
    }
    print(f"  Selected alpha={best_alpha} | Test Macro-F1: {models_results['Complement Naive Bayes']['test_metrics']['macro_f1']:.4f}")

    # 3. Linear SVM (LinearSVC) (Hyperparameter Tuning: C in [0.1, 0.5, 1.0])
    print("\n---> Model 3: Linear SVM (Tuning C in [0.1, 0.5, 1.0])...")
    svm_grid = [0.1, 0.5, 1.0]
    best_svm_score = -1.0
    best_svm_model = None
    best_svm_c = None
    svm_tuning = {}

    for c in svm_grid:
        clf = LinearSVC(C=c, max_iter=2000, random_state=42)
        clf.fit(X_train, train_labels)
        preds = clf.predict(X_val)
        score = f1_score(raw_val_labels, preds, average="macro", zero_division=0)
        svm_tuning[f"C_{c}"] = round(float(score), 4)
        print(f"  LinearSVC C={c} -> Val Macro-F1: {score:.4f}")
        if score > best_svm_score:
            best_svm_score = score
            best_svm_model = clf
            best_svm_c = c

    tuning_log["LinearSVC"] = svm_tuning
    t_start = time.perf_counter()
    best_svm_model.fit(X_train, train_labels)
    svm_fit_time = time.perf_counter() - t_start
    svm_val_preds = best_svm_model.predict(X_val)
    svm_test_preds = best_svm_model.predict(X_test)

    models_results["Linear SVM"] = {
        "best_params": {"C": best_svm_c, "max_iter": 2000},
        "val_metrics": evaluate_predictions(raw_val_labels, svm_val_preds, label_names, class_tiers),
        "test_metrics": evaluate_predictions(raw_test_labels, svm_test_preds, label_names, class_tiers),
        "fit_time_sec": round(svm_fit_time, 2),
    }
    print(f"  Selected C={best_svm_c} | Test Macro-F1: {models_results['Linear SVM']['test_metrics']['macro_f1']:.4f}")

    # 4. Logistic Regression (Hyperparameter Tuning: C in [0.5, 1.0, 5.0])
    print("\n---> Model 4: TF-IDF + Logistic Regression (Tuning C in [0.5, 1.0, 5.0])...")
    lr_grid = [0.5, 1.0, 5.0]
    best_lr_score = -1.0
    best_lr_model = None
    best_lr_c = None
    lr_tuning = {}

    for c in lr_grid:
        clf = LogisticRegression(C=c, max_iter=200, random_state=42, solver="lbfgs")
        clf.fit(X_train, train_labels)
        preds = clf.predict(X_val)
        score = f1_score(raw_val_labels, preds, average="macro", zero_division=0)
        lr_tuning[f"C_{c}"] = round(float(score), 4)
        print(f"  Logistic Regression C={c} -> Val Macro-F1: {score:.4f}")
        if score > best_lr_score:
            best_lr_score = score
            best_lr_model = clf
            best_lr_c = c

    tuning_log["LogisticRegression"] = lr_tuning
    t_start = time.perf_counter()
    best_lr_model.fit(X_train, train_labels)
    lr_fit_time = time.perf_counter() - t_start
    lr_val_preds = best_lr_model.predict(X_val)
    lr_test_preds = best_lr_model.predict(X_test)

    models_results["Logistic Regression"] = {
        "best_params": {"C": best_lr_c, "max_iter": 200, "solver": "lbfgs"},
        "val_metrics": evaluate_predictions(raw_val_labels, lr_val_preds, label_names, class_tiers),
        "test_metrics": evaluate_predictions(raw_test_labels, lr_test_preds, label_names, class_tiers),
        "fit_time_sec": round(lr_fit_time, 2),
    }
    print(f"  Selected C={best_lr_c} | Test Macro-F1: {models_results['Logistic Regression']['test_metrics']['macro_f1']:.4f}")

    # Persist results
    output_payload = {
        "metadata": {
            "dataset": "LEDGAR (LexGLUE)",
            "train_samples_deduped": len(train_texts),
            "val_samples": len(val_texts),
            "test_samples": len(test_texts),
            "features": "TF-IDF (25k max_features, ngram 1-2, sublinear_tf)",
            "random_seed": 42
        },
        "tuning_validation_macro_f1": tuning_log,
        "models": models_results,
    }

    with open(RESULTS_DIR / "baselines_results.json", "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    # [4/5] Generate Visualizations
    print("\n[4/5] Generating baseline evaluation plots...")

    # Plot 1: Test Set Performance Comparison
    model_names = list(models_results.keys())
    test_macros = [models_results[m]["test_metrics"]["macro_f1"] for m in model_names]
    test_micros = [models_results[m]["test_metrics"]["micro_f1"] for m in model_names]
    test_accs = [models_results[m]["test_metrics"]["accuracy"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, test_macros, width, label="Macro-F1 (Headline)", color="#2ca02c")
    plt.bar(x, test_micros, width, label="Micro-F1", color="#1f77b4")
    plt.bar(x + width, test_accs, width, label="Accuracy", color="#ff7f0e")

    plt.xticks(x, model_names, fontsize=10, fontweight="bold")
    plt.ylabel("Score on Test Split (10,000 clauses)", fontsize=11)
    plt.title("Classical Baseline Comparison on LEDGAR Test Split", fontsize=13, fontweight="bold", pad=12)
    plt.ylim(0, 1.0)
    plt.legend(frameon=True, loc="upper left")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(len(model_names)):
        plt.text(x[i] - width, test_macros[i] + 0.02, f"{test_macros[i]:.3f}", ha="center", fontsize=8, fontweight="bold")
        plt.text(x[i], test_micros[i] + 0.02, f"{test_micros[i]:.3f}", ha="center", fontsize=8)
        plt.text(x[i] + width, test_accs[i] + 0.02, f"{test_accs[i]:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "baselines_performance_comparison.png", dpi=300)
    plt.close()

    # Plot 2: Head vs Mid vs Tail Macro-F1 Analysis
    plt.figure(figsize=(10, 6))
    head_f1s = [models_results[m]["test_metrics"]["tier_f1"]["head_f1 (ranks 1-15)"] for m in model_names]
    mid_f1s = [models_results[m]["test_metrics"]["tier_f1"]["mid_f1 (ranks 16-50)"] for m in model_names]
    tail_f1s = [models_results[m]["test_metrics"]["tier_f1"]["tail_f1 (ranks 51-100)"] for m in model_names]

    plt.bar(x - width, head_f1s, width, label="Head (Ranks 1-15, ~48% corpus)", color="#1f77b4")
    plt.bar(x, mid_f1s, width, label="Mid (Ranks 16-50, ~34% corpus)", color="#ff7f0e")
    plt.bar(x + width, tail_f1s, width, label="Tail (Ranks 51-100, ~18% corpus)", color="#d62728")

    plt.xticks(x, model_names, fontsize=10, fontweight="bold")
    plt.ylabel("Macro-F1 within Tier", fontsize=11)
    plt.title("Head vs. Mid vs. Tail Performance Disparity Across Baselines", fontsize=13, fontweight="bold", pad=12)
    plt.ylim(0, 1.0)
    plt.legend(frameon=True, loc="upper left")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(len(model_names)):
        plt.text(x[i] - width, head_f1s[i] + 0.02, f"{head_f1s[i]:.2f}", ha="center", fontsize=8)
        plt.text(x[i], mid_f1s[i] + 0.02, f"{mid_f1s[i]:.2f}", ha="center", fontsize=8)
        plt.text(x[i] + width, tail_f1s[i] + 0.02, f"{tail_f1s[i]:.2f}", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "baselines_head_tail_f1.png", dpi=300)
    plt.close()

    print("\n[5/5] Baselines suite completed successfully!")
    print(f"Results saved to: {RESULTS_DIR / 'baselines_results.json'}")
    print(f"Figures saved to: {FIGURES_DIR}/")
    return models_results


if __name__ == "__main__":
    run_baselines()
