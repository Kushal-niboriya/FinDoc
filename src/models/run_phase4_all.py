"""
Master runner for Phase 4: Neural & Transformer Models.
Executes:
1. FastText EmbeddingBag & TextCNN (PyTorch, GPU)
2. Frozen SBERT (all-MiniLM-L6-v2) + Logistic Regression
3. DistilBERT Fine-Tuning (3 epochs, fp16, GPU)
4. Imbalance Loss Ablation (Cross-Entropy, Balanced Weights, Focal Loss)
5. Generates comparison tables and publication plots.
Outputs:
- results/neural_results.json
- reports/figures/neural_models_comparison.png
- reports/figures/neural_training_curves.png
"""

import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.neural import run_neural_experiments
from src.models.sbert_model import run_sbert_experiment
from src.models.transformer_train import train_distilbert
from src.models.imbalance_ablation import run_imbalance_ablation

FIGURES_DIR = Path("reports/figures")
RESULTS_DIR = Path("results")


def run_all_phase4():
    print("=" * 80)
    print("PHASE 4: MASTER RUNNER FOR NEURAL & TRANSFORMER EXPERIMENTS")
    print("=" * 80)

    all_neural_results = {}

    # Step 1: FastText and TextCNN
    print("\n>>> STEP 1: Running FastText and TextCNN experiments...")
    neural_res = run_neural_experiments()
    all_neural_results.update(neural_res)

    # Step 2: Frozen SBERT + Logistic Regression
    print("\n>>> STEP 2: Running Frozen SBERT (all-MiniLM-L6-v2) experiment...")
    sbert_res = run_sbert_experiment()
    all_neural_results["Frozen SBERT (MiniLM) + LR"] = sbert_res

    # Step 3: DistilBERT Fine-Tuning
    print("\n>>> STEP 3: Running DistilBERT fine-tuning (3 epochs, fp16)...")
    distilbert_res = train_distilbert()
    all_neural_results["DistilBERT (Fine-Tuned)"] = distilbert_res

    # Step 4: Imbalance Ablation
    print("\n>>> STEP 4: Running Imbalance loss ablation...")
    imbalance_res = run_imbalance_ablation()

    # Step 5: Save consolidated results
    payload = {
        "models": all_neural_results,
        "imbalance_ablation": imbalance_res
    }
    with open(RESULTS_DIR / "neural_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Step 6: Generate Visualizations
    print("\n>>> STEP 5: Generating consolidated comparative figures...")

    # Plot 1: Neural & Transformer Models Comparison
    plt.figure(figsize=(12, 6))
    models_to_plot = ["FastText EmbeddingBag", "TextCNN", "Frozen SBERT (MiniLM) + LR", "DistilBERT (Fine-Tuned)"]
    macro_scores = [all_neural_results[m]["test_metrics"]["macro_f1"] for m in models_to_plot]
    micro_scores = [all_neural_results[m]["test_metrics"]["micro_f1"] for m in models_to_plot]
    acc_scores = [all_neural_results[m]["test_metrics"]["accuracy"] for m in models_to_plot]

    x = np.arange(len(models_to_plot))
    width = 0.25

    plt.bar(x - width, macro_scores, width, label="Macro-F1 (Headline)", color="#2ca02c")
    plt.bar(x, micro_scores, width, label="Micro-F1", color="#1f77b4")
    plt.bar(x + width, acc_scores, width, label="Accuracy", color="#ff7f0e")

    plt.xticks(x, [m.replace(" ", "\n") for m in models_to_plot], fontsize=10, fontweight="bold")
    plt.ylabel("Score on Test Split (10,000 clauses)", fontsize=11)
    plt.title("Neural and Transformer Performance Comparison on LEDGAR Test Split", fontsize=13, fontweight="bold", pad=12)
    plt.ylim(0, 1.0)
    plt.legend(frameon=True, loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(len(models_to_plot)):
        plt.text(x[i] - width, macro_scores[i] + 0.02, f"{macro_scores[i]:.4f}", ha="center", fontsize=8, fontweight="bold")
        plt.text(x[i], micro_scores[i] + 0.02, f"{micro_scores[i]:.4f}", ha="center", fontsize=8)
        plt.text(x[i] + width, acc_scores[i] + 0.02, f"{acc_scores[i]:.4f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "neural_models_comparison.png", dpi=300)
    plt.close()

    # Plot 2: DistilBERT Training & Validation Curves
    if "epoch_logs" in distilbert_res:
        plt.figure(figsize=(10, 5))
        epochs = [log["epoch"] for log in distilbert_res["epoch_logs"]]
        train_losses = [log["train_loss"] for log in distilbert_res["epoch_logs"]]
        val_losses = [log["val_loss"] for log in distilbert_res["epoch_logs"]]
        val_f1s = [log["val_macro_f1"] for log in distilbert_res["epoch_logs"]]

        fig, ax1 = plt.subplots(figsize=(9, 5))
        color = "#1f77b4"
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Loss", color=color, fontsize=11)
        l1 = ax1.plot(epochs, train_losses, "o-", color="#1f77b4", label="Train Loss", lw=2)
        l2 = ax1.plot(epochs, val_losses, "s--", color="#ff7f0e", label="Val Loss", lw=2)
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_xticks(epochs)

        ax2 = ax1.twinx()
        color = "#2ca02c"
        ax2.set_ylabel("Validation Macro-F1", color=color, fontsize=11)
        l3 = ax2.plot(epochs, val_f1s, "^-", color="#2ca02c", label="Val Macro-F1", lw=2.5)
        ax2.tick_params(axis="y", labelcolor=color)

        lines = l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="center right", frameon=True)
        plt.title("DistilBERT Training Dynamics & Validation Convergence", fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "neural_training_curves.png", dpi=300)
        plt.close()

    print(f"\nAll Phase 4 experiments completed! Saved to {RESULTS_DIR / 'neural_results.json'}")
    return all_neural_results


if __name__ == "__main__":
    run_all_phase4()
