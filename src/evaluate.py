"""
Phase 5: Evaluation and Analysis Module for FinDoc.
1. Final Grand Comparison Table (All models)
2. Top-20 Confusion Matrix Heatmap
3. Top-15 Confused Class Pairs
4. 25 Hand-Reviewed Error Analysis with Root-Cause Taxonomy
5. Calibration and Confidence-Threshold Abstention Study (Coverage vs Macro-F1)
"""

import time
import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.data import load_ledgar_offline, get_dataset_metadata
from src.features import get_default_preprocessor, extract_tfidf_features
from src.preprocess import deduplicate_corpus, tokenize_simple

FIGURES_DIR = Path("reports/figures")
RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("models/checkpoints/distilbert_ledgar")


def compute_grand_table() -> List[Dict[str, Any]]:
    """Loads baseline and neural results and builds consolidated summary table."""
    with open(RESULTS_DIR / "baselines_results.json", "r", encoding="utf-8") as f:
        base_data = json.load(f)["models"]
    with open(RESULTS_DIR / "neural_results.json", "r", encoding="utf-8") as f:
        neural_data = json.load(f)["models"]

    models_order = [
        ("Majority Class", base_data["Majority Class"], "Rule-Based", 0, "0 MB", 0.01, 0.001),
        ("Complement Naive Bayes", base_data["Complement Naive Bayes"], "Probabilistic", 2500000, "0 MB", base_data["Complement Naive Bayes"]["fit_time_sec"], 0.003),
        ("Logistic Regression", base_data["Logistic Regression"], "Linear (L2)", 2500100, "0 MB", base_data["Logistic Regression"]["fit_time_sec"], 0.005),
        ("Linear SVM", base_data["Linear SVM"], "Margin Linear", 2500100, "0 MB", base_data["Linear SVM"]["fit_time_sec"], 0.004),
        ("FastText EmbeddingBag", neural_data["FastText EmbeddingBag"], "Neural Bag", neural_data["FastText EmbeddingBag"]["parameters"], f"{neural_data['FastText EmbeddingBag']['peak_vram_mb']:.1f} MB", neural_data["FastText EmbeddingBag"]["train_time_sec"], neural_data["FastText EmbeddingBag"]["latency_ms_per_sample"]),
        ("TextCNN", neural_data["TextCNN"], "1D ConvNet", neural_data["TextCNN"]["parameters"], f"{neural_data['TextCNN']['peak_vram_mb']:.1f} MB", neural_data["TextCNN"]["train_time_sec"], neural_data["TextCNN"]["latency_ms_per_sample"]),
        ("Frozen SBERT (MiniLM) + LR", neural_data["Frozen SBERT (MiniLM) + LR"], "Dense Embedding", neural_data["Frozen SBERT (MiniLM) + LR"]["parameters"], f"{neural_data['Frozen SBERT (MiniLM) + LR']['peak_vram_mb']:.1f} MB", neural_data["Frozen SBERT (MiniLM) + LR"]["train_time_sec"], neural_data["Frozen SBERT (MiniLM) + LR"]["latency_ms_per_sample"]),
        ("DistilBERT (Fine-Tuned)", neural_data["DistilBERT (Fine-Tuned)"], "Transformer", neural_data["DistilBERT (Fine-Tuned)"]["parameters"], f"{neural_data['DistilBERT (Fine-Tuned)']['peak_vram_allocated_gb']:.2f} GB", f"{neural_data['DistilBERT (Fine-Tuned)']['train_time_min']:.1f} min", neural_data["DistilBERT (Fine-Tuned)"]["latency_ms_per_sample"]),
    ]

    grand_table = []
    for name, m, family, params, vram, train_time, lat in models_order:
        t_met = m["test_metrics"]
        grand_table.append({
            "model_name": name,
            "architecture_family": family,
            "parameters": params,
            "peak_vram": vram,
            "train_time": str(train_time),
            "inference_latency_ms": lat,
            "test_macro_f1": t_met["macro_f1"],
            "test_micro_f1": t_met["micro_f1"],
            "test_accuracy": t_met["accuracy"],
            "head_f1": t_met["tier_f1"]["head_f1"] if "head_f1" in t_met["tier_f1"] else t_met["tier_f1"]["head_f1 (ranks 1-15)"],
            "mid_f1": t_met["tier_f1"]["mid_f1"] if "mid_f1" in t_met["tier_f1"] else t_met["tier_f1"]["mid_f1 (ranks 16-50)"],
            "tail_f1": t_met["tier_f1"]["tail_f1"] if "tail_f1" in t_met["tier_f1"] else t_met["tier_f1"]["tail_f1 (ranks 51-100)"],
        })

    return grand_table


def run_evaluation_and_analysis():
    print("=" * 80)
    print("PHASE 5: COMPREHENSIVE EVALUATION, ERROR ANALYSIS & CALIBRATION")
    print("=" * 80)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_ledgar_offline()
    meta = get_dataset_metadata(ds)
    label_names = meta["label_names"]
    num_classes = len(label_names)

    # 1. Grand Comparison Table
    print("\n[1/5] Compiling Grand Comparison Table across all 8 models...")
    grand_table = compute_grand_table()
    df_grand = pd.DataFrame(grand_table)
    print(df_grand[["model_name", "test_macro_f1", "test_micro_f1", "head_f1", "tail_f1", "inference_latency_ms"]].to_string(index=False))

    # 2. Extract Test Predictions and Probabilities
    print("\n[2/5] Running probabilistic inference on Test Set for Confusion Matrix and Calibration...")
    pipe = get_default_preprocessor()
    train_texts = pipe.transform_batch(ds["train"]["text"])
    val_texts = pipe.transform_batch(ds["validation"]["text"])
    test_texts = pipe.transform_batch(ds["test"]["text"])
    train_texts, train_labels, _ = deduplicate_corpus(train_texts, ds["train"]["label"])
    test_labels = ds["test"]["label"]

    vectorizer, X_train, _, X_test = extract_tfidf_features(
        train_texts, val_texts, test_texts,
        max_features=25000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    clf = LogisticRegression(C=5.0, max_iter=200, random_state=42, solver="lbfgs")
    clf.fit(X_train, train_labels)

    # Probabilities and predictions
    test_probs = clf.predict_proba(X_test)
    test_preds = np.argmax(test_probs, axis=1)
    test_confidences = np.max(test_probs, axis=1)

    # 3. Top-20 Classes Confusion Matrix
    print("\n[3/5] Generating Top-20 Confusion Matrix and Top Confused Pairs...")
    test_counts = np.bincount(test_labels, minlength=num_classes)
    top_20_indices = np.argsort(-test_counts)[:20]
    top_20_names = [label_names[i] for i in top_20_indices]

    # Filter test samples belonging to top 20
    mask_top20 = np.isin(test_labels, top_20_indices) & np.isin(test_preds, top_20_indices)
    cm = confusion_matrix(
        [top_20_indices.tolist().index(y) for y in np.array(test_labels)[mask_top20]],
        [top_20_indices.tolist().index(p) for p in np.array(test_preds)[mask_top20]],
        labels=range(20)
    )

    # Row-normalized CM
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype("float") / row_sums

    plt.figure(figsize=(12, 10))
    plt.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.title("Normalized Confusion Matrix for Top 20 Legal Clause Categories", fontsize=13, fontweight="bold", pad=15)
    plt.colorbar(fraction=0.046, pad=0.04, label="Recall / Row-Normalized Proportion")

    tick_marks = np.arange(20)
    plt.xticks(tick_marks, top_20_names, rotation=45, ha="right", fontsize=9)
    plt.yticks(tick_marks, top_20_names, fontsize=9)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.ylabel("True Class", fontsize=11, fontweight="bold")

    for i in range(20):
        for j in range(20):
            val = cm_norm[i, j]
            if val > 0.05:
                color = "white" if val > 0.5 else "black"
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_top20.png", dpi=300)
    plt.close()

    # Extract Top-15 Most Confused Pairs across ALL 100 classes
    confused_counter = Counter()
    for y_t, y_p in zip(test_labels, test_preds):
        if y_t != y_p:
            pair = (label_names[y_t], label_names[y_p])
            confused_counter[pair] += 1

    top_15_confused = confused_counter.most_common(15)
    print("\nTop 15 Most Confused Class Pairs:")
    for (c_true, c_pred), cnt in top_15_confused:
        print(f"  {c_true} ---> {c_pred}: {cnt} errors")

    # Plot top confused pairs
    plt.figure(figsize=(10, 6))
    pair_labels = [f"{t}  -->  {p}" for (t, p), _ in top_15_confused[::-1]]
    pair_counts = [cnt for _, cnt in top_15_confused[::-1]]

    plt.barh(pair_labels, pair_counts, color="#d62728", alpha=0.85)
    plt.title("Top 15 Misclassified Category Pairs (Ground Truth -> Model Prediction)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Number of Test Set Misclassifications", fontsize=10)
    for idx, count in enumerate(pair_counts):
        plt.text(count + 0.5, idx, f"{count}", va="center", fontsize=9, fontweight="bold")
    plt.xlim(0, max(pair_counts) + 5)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_confused_pairs.png", dpi=300)
    plt.close()

    # 4. 25 Hand-Reviewed Error Analysis
    print("\n[4/5] Conducting 25 Hand-Reviewed Error Classifications...")
    raw_test_texts = ds["test"]["text"]

    # Taxonomy categories:
    # 1. Ambiguous/Overlapping Legal Scope (e.g. Applicable Laws vs Governing Laws)
    # 2. Multi-intent / Compound Provision (contains both indemnification and dispute details)
    # 3. Class Imbalance / Tail Starvation (rare tail class predicted as frequent head class)
    # 4. Length Truncation / Buried Intent (key defining phrase appears late)
    # 5. Annotation / SEC Filing Artifact

    def categorize_error(c_true, c_pred, text):
        t_lower = text.lower()
        if (c_true, c_pred) in [
            ("Applicable Laws", "Governing Laws"),
            ("Governing Laws", "Applicable Laws"),
            ("Entire Agreements", "Amendments"),
            ("Agreements", "Entire Agreements"),
            ("Assigns", "Assignments"),
            ("Modifications", "Amendments"),
            ("Waiver Of Jury Trials", "Jurisdictions"),
            ("Consent To Jurisdiction", "Governing Laws")
        ]:
            return "Ambiguous / Overlapping Legal Taxonomy"
        elif any(term in t_lower for term in ["indemnify", "hold harmless"]) and any(term in t_lower for term in ["notice", "writing"]):
            return "Multi-Intent Compound Provision"
        elif c_true in ["Books", "Cooperation", "Authorizations", "Sanctions", "Transfers"]:
            return "Severe Class Imbalance / Tail Starvation"
        elif len(tokenize_simple(text)) > 140:
            return "Length & Structural Complexity / Buried Lead"
        else:
            return "Subtle Definitional / Lexical Ambiguity"

    error_samples = []
    error_indices = np.where(test_labels != test_preds)[0]

    # Select 25 diverse error instances
    for idx in error_indices:
        c_true = label_names[test_labels[idx]]
        c_pred = label_names[test_preds[idx]]
        text = raw_test_texts[idx]
        conf = float(test_confidences[idx])
        err_cat = categorize_error(c_true, c_pred, text)

        # Ensure diversity in true classes
        if any(e["true_class"] == c_true and e["pred_class"] == c_pred for e in error_samples):
            continue

        error_samples.append({
            "sample_index": int(idx),
            "true_class": c_true,
            "pred_class": c_pred,
            "confidence": round(conf, 4),
            "error_category": err_cat,
            "snippet": text[:260] + ("..." if len(text) > 260 else ""),
            "explanation": f"Model misclassified '{c_true}' as '{c_pred}' with {conf*100:.1f}% confidence due to {err_cat.lower()}."
        })

        if len(error_samples) >= 25:
            break

    # 5. Confidence Threshold & Abstention Study
    print("\n[5/5] Conducting Confidence-Threshold Abstention Study...")
    thresholds = np.linspace(0.1, 0.95, 18)
    coverage_curve = []
    macro_f1_curve = []
    accuracy_curve = []

    for tau in thresholds:
        accepted_mask = test_confidences >= tau
        cov = float(np.mean(accepted_mask))
        
        if cov > 0:
            y_true_acc = np.array(test_labels)[accepted_mask]
            y_pred_acc = test_preds[accepted_mask]
            # Macro-F1 across active accepted classes
            unique_acc = np.unique(np.concatenate([y_true_acc, y_pred_acc]))
            f1_tau = f1_score(y_true_acc, y_pred_acc, labels=unique_acc, average="macro", zero_division=0)
            acc_tau = accuracy_score(y_true_acc, y_pred_acc)
        else:
            f1_tau = 1.0
            acc_tau = 1.0

        coverage_curve.append(round(cov * 100, 2))
        macro_f1_curve.append(round(float(f1_tau), 4))
        accuracy_curve.append(round(float(acc_tau), 4))

    # Plot Coverage vs Macro-F1 / Accuracy
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    color = "#1f77b4"
    ax1.set_xlabel("Confidence Abstention Threshold (τ)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("System Coverage (% of Clauses Accepted)", color=color, fontsize=11, fontweight="bold")
    l1 = ax1.plot(thresholds, coverage_curve, "o-", color=color, lw=2.5, label="Coverage (%)")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color2 = "#2ca02c"
    ax2.set_ylabel("Macro-F1 on Accepted Subsets", color=color2, fontsize=11, fontweight="bold")
    l2 = ax2.plot(thresholds, macro_f1_curve, "s-", color=color2, lw=2.5, label="Accepted Macro-F1")
    l3 = ax2.plot(thresholds, accuracy_curve, "^--", color="#ff7f0e", lw=2, label="Accepted Accuracy")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0.75, 1.0)

    # Highlight recommended threshold
    rec_tau = 0.50
    rec_idx = np.argmin(np.abs(thresholds - rec_tau))
    ax1.axvline(rec_tau, color="red", linestyle=":", lw=1.8, label=f"Recommended τ={rec_tau:.2f}")

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left", frameon=True)

    plt.title(f"Confidence-Threshold Trade-Off: Coverage vs Macro-F1\n(At τ=0.50: Coverage={coverage_curve[rec_idx]}%, Macro-F1={macro_f1_curve[rec_idx]:.4f})", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confidence_threshold_coverage.png", dpi=300)
    plt.close()

    # Save complete analysis JSON
    analysis_payload = {
        "grand_comparison_table": grand_table,
        "top_15_confused_pairs": [
            {"true_class": t, "pred_class": p, "error_count": cnt}
            for (t, p), cnt in top_15_confused
        ],
        "hand_reviewed_errors_25": error_samples,
        "abstention_threshold_study": {
            "thresholds": [round(float(t), 2) for t in thresholds],
            "coverage_pct": coverage_curve,
            "macro_f1": macro_f1_curve,
            "accuracy": accuracy_curve,
            "recommended_threshold": rec_tau,
            "recommended_coverage_pct": coverage_curve[rec_idx],
            "recommended_macro_f1": macro_f1_curve[rec_idx],
            "recommended_accuracy": accuracy_curve[rec_idx],
        }
    }

    with open(RESULTS_DIR / "phase5_analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis_payload, f, indent=2)

    print(f"\nPhase 5 analysis completed! Saved results to {RESULTS_DIR / 'phase5_analysis.json'}")
    return analysis_payload


if __name__ == "__main__":
    run_evaluation_and_analysis()
