"""
Imbalance Ablation Study for FinDoc:
Compares loss weighting and sampling strategies on severe legal class imbalance:
1. Unweighted Cross-Entropy (Baseline)
2. Inverse-Frequency Class Weights (Balanced Weights)
3. Focal Loss (Lin et al., 2017; gamma=2.0)
4. Random Oversampling of Tail Classes

Analyzes trade-offs between Head accuracy and Tail Macro-F1.
Outputs:
- results/imbalance_ablation.json
- reports/figures/imbalance_ablation_tail_f1.png
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from src.data import load_ledgar_offline, get_dataset_metadata
from src.features import get_default_preprocessor
from src.preprocess import deduplicate_corpus

FIGURES_DIR = Path("reports/figures")
RESULTS_DIR = Path("results")


class FocalLoss(nn.Module):
    """Multi-class Focal Loss (Lin et al., 2017)."""
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class DenseClassifier(nn.Module):
    """Simple linear probe on top of 384-d dense embeddings."""
    def __init__(self, in_features: int = 384, num_classes: int = 100):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


def run_imbalance_ablation():
    print("=" * 70)
    print("PHASE 4: IMBALANCE MITIGATION ABLATION STUDY")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = load_ledgar_offline()
    meta = get_dataset_metadata(ds)
    label_names = meta["label_names"]
    num_classes = len(label_names)

    # Class tiers
    train_counts = np.bincount(ds["train"]["label"], minlength=num_classes)
    sorted_classes = np.argsort(-train_counts)
    class_tiers = {
        "head": sorted_classes[:15].tolist(),
        "mid": sorted_classes[15:50].tolist(),
        "tail": sorted_classes[50:].tolist(),
    }

    pipe = get_default_preprocessor()
    print("Preprocessing texts...")
    train_texts = pipe.transform_batch(ds["train"]["text"])
    val_texts = pipe.transform_batch(ds["validation"]["text"])
    test_texts = pipe.transform_batch(ds["test"]["text"])

    train_texts, train_labels, _ = deduplicate_corpus(train_texts, ds["train"]["label"])
    val_labels = ds["validation"]["label"]
    test_labels = ds["test"]["label"]

    print("Encoding with sentence-transformers/all-MiniLM-L6-v2 on GPU...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(device))
    if device.type == "cuda":
        encoder.half()

    X_train_np = encoder.encode(train_texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    X_val_np = encoder.encode(val_texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    X_test_np = encoder.encode(test_texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    y_val = torch.tensor(val_labels, dtype=torch.long)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.long)

    # Compute balanced class weights
    classes = np.arange(num_classes)
    weights_np = compute_class_weight("balanced", classes=classes, y=np.array(train_labels))
    # Cap maximum weight to prevent extreme gradient instability on rare classes with < 30 samples
    weights_np = np.clip(weights_np, a_min=0.1, a_max=20.0)
    class_weights_tensor = torch.tensor(weights_np, dtype=torch.float32).to(device)

    # Setup configurations
    strategies = {
        "1. Unweighted Cross-Entropy": {
            "loss_fn": nn.CrossEntropyLoss(),
            "oversample": False
        },
        "2. Balanced Class Weights": {
            "loss_fn": nn.CrossEntropyLoss(weight=class_weights_tensor),
            "oversample": False
        },
        "3. Focal Loss (gamma=2.0)": {
            "loss_fn": FocalLoss(gamma=2.0),
            "oversample": False
        },
        "4. Focal Loss + Class Weights": {
            "loss_fn": FocalLoss(gamma=2.0, weight=class_weights_tensor),
            "oversample": False
        }
    }

    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256, shuffle=False)

    results = []
    print("\nTraining linear probes under each imbalance loss formulation...")

    for name, cfg in strategies.items():
        print(f"\n---> Evaluating: {name}...")
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)

        model = DenseClassifier(in_features=384, num_classes=num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = cfg["loss_fn"]

        best_val_macro = -1.0
        best_state = None

        t_start = time.perf_counter()
        for epoch in range(1, 11):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = model(bx)
                loss = loss_fn(logits, by)
                loss.backward()
                optimizer.step()

            # Val check
            model.eval()
            val_preds = []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx = bx.to(device)
                    logits = model(bx)
                    val_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())

            labels_all = list(range(num_classes))
            val_macro = f1_score(val_labels, val_preds, labels=labels_all, average="macro", zero_division=0)
            if val_macro > best_val_macro:
                best_val_macro = val_macro
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        fit_time = time.perf_counter() - t_start

        # Test evaluation with best model
        model.load_state_dict(best_state)
        model.eval()
        test_preds = []
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(device)
                logits = model(bx)
                test_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())

        test_macro = f1_score(test_labels, test_preds, labels=labels_all, average="macro", zero_division=0)
        test_micro = f1_score(test_labels, test_preds, labels=labels_all, average="micro", zero_division=0)
        test_acc = accuracy_score(test_labels, test_preds)
        per_class_f1 = f1_score(test_labels, test_preds, labels=labels_all, average=None, zero_division=0)

        head_f1 = float(np.mean([per_class_f1[i] for i in class_tiers["head"]]))
        mid_f1 = float(np.mean([per_class_f1[i] for i in class_tiers["mid"]]))
        tail_f1 = float(np.mean([per_class_f1[i] for i in class_tiers["tail"]]))

        print(f"  Result -> Macro-F1: {test_macro:.4f} | Micro-F1: {test_micro:.4f} | Head: {head_f1:.4f} | Mid: {mid_f1:.4f} | Tail: {tail_f1:.4f}")

        results.append({
            "strategy": name,
            "val_macro_f1": round(float(best_val_macro), 4),
            "test_macro_f1": round(float(test_macro), 4),
            "test_micro_f1": round(float(test_micro), 4),
            "test_accuracy": round(float(test_acc), 4),
            "tier_f1": {
                "head_f1": round(head_f1, 4),
                "mid_f1": round(mid_f1, 4),
                "tail_f1": round(tail_f1, 4),
            },
            "fit_time_sec": round(fit_time, 2)
        })

    # Save to JSON
    with open(RESULTS_DIR / "imbalance_ablation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Plotting Head vs Mid vs Tail under each strategy
    print("\nGenerating imbalance ablation plot...")
    plt.figure(figsize=(11, 6))
    names = [r["strategy"] for r in results]
    head_vals = [r["tier_f1"]["head_f1"] for r in results]
    mid_vals = [r["tier_f1"]["mid_f1"] for r in results]
    tail_vals = [r["tier_f1"]["tail_f1"] for r in results]
    macro_vals = [r["test_macro_f1"] for r in results]

    x = np.arange(len(names))
    width = 0.2

    plt.bar(x - 1.5 * width, head_vals, width, label="Head F1 (Top 15)", color="#1f77b4")
    plt.bar(x - 0.5 * width, mid_vals, width, label="Mid F1 (Ranks 16-50)", color="#ff7f0e")
    plt.bar(x + 0.5 * width, tail_vals, width, label="Tail F1 (Ranks 51-100)", color="#d62728")
    plt.bar(x + 1.5 * width, macro_vals, width, label="Overall Macro-F1", color="#2ca02c")

    plt.xticks(x, [n.replace(" ", "\n") for n in names], fontsize=9, fontweight="bold")
    plt.ylabel("Score on Test Split", fontsize=11)
    plt.title("Impact of Loss Weighting and Focal Loss on Rare Legal Provisions (Tail)", fontsize=13, fontweight="bold", pad=12)
    plt.ylim(0.4, 0.95)
    plt.legend(frameon=True, loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(len(names)):
        plt.text(x[i] + 0.5 * width, tail_vals[i] + 0.015, f"{tail_vals[i]:.3f}", ha="center", fontsize=8, fontweight="bold", color="darkred")
        plt.text(x[i] + 1.5 * width, macro_vals[i] + 0.015, f"{macro_vals[i]:.3f}", ha="center", fontsize=8, fontweight="bold", color="darkgreen")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "imbalance_ablation_tail_f1.png", dpi=300)
    plt.close()

    print(f"Saved imbalance ablation results to {RESULTS_DIR / 'imbalance_ablation.json'} and plot to {FIGURES_DIR / 'imbalance_ablation_tail_f1.png'}")
    return results


if __name__ == "__main__":
    run_imbalance_ablation()
