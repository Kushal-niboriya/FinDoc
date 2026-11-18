"""
Phase 1: Exploratory Data Analysis (EDA) for FinDoc.
Computes dataset statistics, class distributions, token-length distributions,
vocabulary and n-gram analyses, duplicate/leakage quantification, and generates plots.
Outputs:
- results/eda_stats.json
- reports/figures/*.png
"""

import os
import re
import json
import hashlib
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from src.data import load_ledgar_offline, get_dataset_metadata

FIGURES_DIR = Path("reports/figures")
RESULTS_DIR = Path("results")

# Common English stopwords to enhance n-gram and wordcloud clarity
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been before being below
between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during
each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's
hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's
me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out
over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their
theirs them themselves then there there's these they they'd they'll they're they've this those through
to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's
where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're
you've your yours yourself yourselves
""".split())


def tokenize_simple(text: str) -> List[str]:
    """Fast word-regex tokenization."""
    return re.findall(r"\b\w+\b", text.lower())


def analyze_dataset():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading LEDGAR dataset offline...")
    ds = load_ledgar_offline()
    meta = get_dataset_metadata(ds)
    label_names = meta["label_names"]
    num_classes = len(label_names)

    train_data = ds["train"]
    val_data = ds["validation"]
    test_data = ds["test"]

    print(f"Loaded: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}, Classes={num_classes}")

    # Convert to efficient pandas structures for analysis
    print("[2/6] Building dataframes and computing token lengths...")
    dfs = {}
    for split in ["train", "validation", "test"]:
        texts = ds[split]["text"]
        labels = ds[split]["label"]
        lengths = [len(tokenize_simple(t)) for t in texts]
        dfs[split] = pd.DataFrame({
            "text": texts,
            "label": labels,
            "label_name": [label_names[l] for l in labels],
            "token_length": lengths
        })

    train_df = dfs["train"]
    val_df = dfs["validation"]
    test_df = dfs["test"]

    # 1. Class Distribution Analysis
    print("[3/6] Analyzing class distribution and long tail...")
    train_counts = train_df["label_name"].value_counts()
    val_counts = val_df["label_name"].value_counts()
    test_counts = test_df["label_name"].value_counts()

    # Sort classes by train frequency
    sorted_classes = train_counts.index.tolist()
    class_stats = []
    for rank, cls in enumerate(sorted_classes, start=1):
        tr_c = int(train_counts.get(cls, 0))
        va_c = int(val_counts.get(cls, 0))
        te_c = int(test_counts.get(cls, 0))
        tot = tr_c + va_c + te_c
        class_stats.append({
            "rank": rank,
            "class_name": cls,
            "train_count": tr_c,
            "val_count": va_c,
            "test_count": te_c,
            "total_count": tot,
            "train_prop": round(tr_c / len(train_df), 5),
            "tier": "Head (1-15)" if rank <= 15 else ("Mid (16-50)" if rank <= 50 else "Tail (51-100)")
        })

    # Class distribution plot (log scale)
    plt.figure(figsize=(14, 6))
    x_indices = np.arange(len(sorted_classes))
    y_train = [train_counts[c] for c in sorted_classes]
    
    plt.bar(x_indices, y_train, color="#1f77b4", edgecolor="none", width=0.8, alpha=0.9)
    plt.yscale("log")
    plt.title("LEDGAR Class Distribution across 100 Clause Categories (Train Split, Log Scale)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Category Rank (1 = Most Frequent, 100 = Least Frequent)", fontsize=11)
    plt.ylabel("Number of Clause Examples (Log Scale)", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    
    # Highlight Head, Mid, Tail
    plt.axvline(15 - 0.5, color="green", linestyle=":", linewidth=1.5, label="Head/Mid Boundary (Top 15)")
    plt.axvline(50 - 0.5, color="red", linestyle=":", linewidth=1.5, label="Mid/Tail Boundary (Rank 50)")
    
    # Annotate top 3 and bottom 3
    plt.annotate(f"#1 {sorted_classes[0]} ({y_train[0]})", xy=(0, y_train[0]), xytext=(3, y_train[0] * 0.8),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.8), fontsize=8)
    plt.annotate(f"#100 {sorted_classes[-1]} ({y_train[-1]})", xy=(99, y_train[-1]), xytext=(70, y_train[-1] * 2),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.8), fontsize=8)
    
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution_log.png", dpi=300)
    plt.close()

    # Cumulative distribution plot (Pareto analysis)
    plt.figure(figsize=(8, 5))
    cum_share = np.cumsum(y_train) / sum(y_train) * 100
    plt.plot(np.arange(1, 101), cum_share, color="#d62728", lw=2.5, marker="o", markersize=3, markevery=5)
    top_10_share = cum_share[9]
    top_20_share = cum_share[19]
    plt.axvline(10, color="gray", linestyle="--", alpha=0.7)
    plt.axhline(top_10_share, color="gray", linestyle="--", alpha=0.7)
    plt.axvline(20, color="gray", linestyle="--", alpha=0.7)
    plt.axhline(top_20_share, color="gray", linestyle="--", alpha=0.7)
    plt.scatter([10, 20], [top_10_share, top_20_share], color="black", zorder=5)
    plt.text(12, top_10_share - 4, f"Top 10 classes = {top_10_share:.1f}%", fontsize=9, fontweight="bold")
    plt.text(22, top_20_share - 4, f"Top 20 classes = {top_20_share:.1f}%", fontsize=9, fontweight="bold")
    plt.title("Cumulative Class Frequency (Pareto Long Tail)", fontsize=12, fontweight="bold")
    plt.xlabel("Number of Top Categories", fontsize=10)
    plt.ylabel("Cumulative Share of Training Examples (%)", fontsize=10)
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution_head_tail.png", dpi=300)
    plt.close()

    # 2. Token Length Statistics per split
    print("[4/6] Computing token-length distributions per split and per class...")
    length_stats = {}
    for split_name, df_s in dfs.items():
        lens = df_s["token_length"].values
        length_stats[split_name] = {
            "count": int(len(lens)),
            "mean": round(float(np.mean(lens)), 2),
            "std": round(float(np.std(lens)), 2),
            "median": int(np.median(lens)),
            "min": int(np.min(lens)),
            "max": int(np.max(lens)),
            "p25": int(np.percentile(lens, 25)),
            "p75": int(np.percentile(lens, 75)),
            "p90": int(np.percentile(lens, 90)),
            "p95": int(np.percentile(lens, 95)),
            "p99": int(np.percentile(lens, 99)),
            "pct_le_128": round(float(np.mean(lens <= 128) * 100), 2),
            "pct_le_256": round(float(np.mean(lens <= 256) * 100), 2),
            "pct_le_512": round(float(np.mean(lens <= 512) * 100), 2),
        }

    # Plot token length histograms per split
    plt.figure(figsize=(10, 5))
    bins = np.linspace(0, 300, 60)
    for split_name, col in [("train", "#1f77b4"), ("validation", "#ff7f0e"), ("test", "#2ca02c")]:
        plt.hist(dfs[split_name]["token_length"], bins=bins, density=True, alpha=0.5, label=f"{split_name.capitalize()} (median={length_stats[split_name]['median']})", color=col)
    
    plt.axvline(128, color="red", linestyle="--", linewidth=1.8, label="Transformer max_len=128 cutoff")
    plt.title("Token Length Distribution Across Splits", fontsize=12, fontweight="bold")
    plt.xlabel("Token Count per Clause", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.xlim(0, 300)
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "token_length_distribution.png", dpi=300)
    plt.close()

    # Token lengths for top 15 classes (boxplot)
    top_15_classes = sorted_classes[:15]
    top_15_lengths = [train_df[train_df["label_name"] == c]["token_length"].values for c in top_15_classes]
    
    plt.figure(figsize=(12, 6))
    box = plt.boxplot(top_15_lengths, patch_artist=True, showfliers=False, vert=True)
    for patch in box['boxes']:
        patch.set_facecolor('#aec7e8')
        patch.set_edgecolor('#1f77b4')
    plt.xticks(range(1, 16), top_15_classes, rotation=45, ha="right", fontsize=9)
    plt.ylabel("Token Length (Whiskers: IQR without fliers)", fontsize=10)
    plt.title("Clause Token Length Variance Across Top 15 Legal Categories", fontsize=12, fontweight="bold")
    plt.axhline(128, color="red", linestyle="--", alpha=0.8, label="128 tokens threshold")
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "token_length_by_class.png", dpi=300)
    plt.close()

    # 3. Vocabulary Size and N-grams
    print("[5/6] Computing vocabulary, top n-grams, and representative word clouds...")
    token_counter = Counter()
    bigram_counter = Counter()
    trigram_counter = Counter()

    for text in train_df["text"]:
        tokens = tokenize_simple(text)
        token_counter.update(tokens)
        
        # Filtered for meaningful legal n-grams
        filtered = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        if len(filtered) >= 2:
            bigrams = [f"{filtered[i]} {filtered[i+1]}" for i in range(len(filtered) - 1)]
            bigram_counter.update(bigrams)
        if len(filtered) >= 3:
            trigrams = [f"{filtered[i]} {filtered[i+1]} {filtered[i+2]}" for i in range(len(filtered) - 2)]
            trigram_counter.update(trigrams)

    total_tokens_train = sum(token_counter.values())
    vocab_size_train = len(token_counter)
    singletons = sum(1 for _, count in token_counter.items() if count == 1)

    # Plot top N-grams
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    top_unigrams = [item for item in token_counter.most_common(50) if item[0] not in STOPWORDS][:12]
    top_bigrams = bigram_counter.most_common(12)
    top_trigrams = trigram_counter.most_common(12)

    for ax, data, title, col in zip(
        axes,
        [top_unigrams, top_bigrams, top_trigrams],
        ["Top Content Unigrams", "Top Content Bigrams", "Top Content Trigrams"],
        ["#1f77b4", "#2ca02c", "#ff7f0e"]
    ):
        labels_ng, counts_ng = zip(*data[::-1])
        ax.barh(range(len(labels_ng)), counts_ng, color=col, alpha=0.85)
        ax.set_yticks(range(len(labels_ng)))
        ax.set_yticklabels(labels_ng, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.suptitle("Most Frequent Informative N-Grams in LEDGAR Training Corpus", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_ngrams.png", dpi=300)
    plt.close()

    # Per-class word clouds for 6 representative classes:
    # 2 Head: 'Governing Law', 'Amendments'
    # 2 Mid: 'Indemnifications' / 'Severability' (check exact names in label_names)
    # 2 Tail: 'Anti-Corruption Laws', 'Adjustments'
    target_candidates = ["Governing Law", "Amendments", "Indemnifications", "Severability", "Anti-Corruption Laws", "Adjustments"]
    chosen_classes = [c for c in target_candidates if c in label_names]
    # If any is missing, pick closest available
    if len(chosen_classes) < 6:
        for c in sorted_classes:
            if c not in chosen_classes:
                chosen_classes.append(c)
            if len(chosen_classes) == 6:
                break

    print(f"Generating word clouds for 6 representative classes: {chosen_classes}")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, cls_name in enumerate(chosen_classes):
        cls_texts = " ".join(train_df[train_df["label_name"] == cls_name]["text"].tolist()[:300])
        wc = WordCloud(
            width=500,
            height=300,
            background_color="white",
            stopwords=STOPWORDS,
            colormap="Blues" if idx < 2 else ("Greens" if idx < 4 else "Reds"),
            max_words=60,
            random_state=42
        ).generate(cls_texts)

        tier_label = "Head" if idx < 2 else ("Mid" if idx < 4 else "Tail")
        axes[idx].imshow(wc, interpolation="bilinear")
        axes[idx].set_title(f"{cls_name} ({tier_label})", fontsize=11, fontweight="bold", pad=8)
        axes[idx].axis("off")

    plt.suptitle("Domain Vocabulary Word Clouds Across 6 Representative Legal Clause Classes", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "wordclouds_representative_classes.png", dpi=300)
    plt.close()

    # 4. Duplicates & Cross-Split Data Leakage Check
    print("[6/6] Checking for exact duplicates and cross-split leakage...")
    
    def hash_str(s: str) -> str:
        # Normalize whitespace and case for robust comparison
        norm = re.sub(r"\s+", " ", s.strip().lower())
        return hashlib.md5(norm.encode("utf-8")).hexdigest()

    hashes_train = [hash_str(t) for t in train_df["text"]]
    hashes_val = [hash_str(t) for t in val_df["text"]]
    hashes_test = [hash_str(t) for t in test_df["text"]]

    set_train = set(hashes_train)
    set_val = set(hashes_val)
    set_test = set(hashes_test)

    train_internal_dups = len(hashes_train) - len(set_train)
    val_internal_dups = len(hashes_val) - len(set_val)
    test_internal_dups = len(hashes_test) - len(set_test)

    # Cross-split overlap (leakage)
    train_val_overlap = set_train.intersection(set_val)
    train_test_overlap = set_train.intersection(set_test)
    val_test_overlap = set_val.intersection(set_test)

    leakage_stats = {
        "within_split_duplicates": {
            "train_duplicates": train_internal_dups,
            "train_duplicate_pct": round(train_internal_dups / len(train_df) * 100, 2),
            "val_duplicates": val_internal_dups,
            "val_duplicate_pct": round(val_internal_dups / len(val_df) * 100, 2),
            "test_duplicates": test_internal_dups,
            "test_duplicate_pct": round(test_internal_dups / len(test_df) * 100, 2),
        },
        "cross_split_leakage": {
            "train_val_shared_unique_texts": len(train_val_overlap),
            "train_val_leakage_pct_of_val": round(len(train_val_overlap) / len(set_val) * 100, 2),
            "train_test_shared_unique_texts": len(train_test_overlap),
            "train_test_leakage_pct_of_test": round(len(train_test_overlap) / len(set_test) * 100, 2),
            "val_test_shared_unique_texts": len(val_test_overlap),
            "val_test_leakage_pct_of_test": round(len(val_test_overlap) / len(set_test) * 100, 2),
        }
    }

    # Consolidated stats object
    eda_stats = {
        "dataset_summary": {
            "name": "LEDGAR (LexGLUE benchmark config)",
            "source": "US SEC EDGAR corporate filings (material contracts)",
            "splits": meta["splits"],
            "split_sizes": meta["split_sizes"],
            "total_samples": meta["total_examples"],
            "fields": meta["field_names"],
            "num_classes": num_classes,
        },
        "class_imbalance": {
            "top_1_class": sorted_classes[0],
            "top_1_count": int(train_counts.iloc[0]),
            "top_100_class": sorted_classes[-1],
            "top_100_count": int(train_counts.iloc[-1]),
            "imbalance_ratio_head_to_tail": round(float(train_counts.iloc[0] / train_counts.iloc[-1]), 2),
            "top_10_classes_cumulative_pct": round(top_10_share, 2),
            "top_20_classes_cumulative_pct": round(top_20_share, 2),
            "class_rankings": class_stats,
        },
        "token_lengths": length_stats,
        "vocabulary": {
            "total_tokens_train": total_tokens_train,
            "unique_vocab_size_train": vocab_size_train,
            "singleton_tokens": singletons,
            "singleton_pct": round(singletons / vocab_size_train * 100, 2),
            "top_20_unigrams": [{"term": t, "count": c} for t, c in top_unigrams[:20]],
            "top_15_bigrams": [{"term": t, "count": c} for t, c in top_bigrams[:15]],
            "top_15_trigrams": [{"term": t, "count": c} for t, c in top_trigrams[:15]],
        },
        "data_leakage_and_duplicates": leakage_stats,
    }

    stats_path = RESULTS_DIR / "eda_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(eda_stats, f, indent=2)

    print("\n" + "=" * 60)
    print("EDA SUMMARY COMPLETED")
    print("=" * 60)
    print(f"Total dataset clauses:      {meta['total_examples']:,}")
    print(f"Number of categories:       {num_classes}")
    print(f"Max class count (Train):    {sorted_classes[0]} = {train_counts.iloc[0]:,}")
    print(f"Min class count (Train):    {sorted_classes[-1]} = {train_counts.iloc[-1]:,}")
    print(f"Imbalance ratio (Max/Min):  {train_counts.iloc[0] / train_counts.iloc[-1]:.2f}x")
    print(f"Top 10 categories account:  {top_10_share:.1f}% of train split")
    print(f"Mean token length:          Train={length_stats['train']['mean']}, Median={length_stats['train']['median']}")
    print(f"Clauses <= 128 tokens:      {length_stats['train']['pct_le_128']}%")
    print(f"Unique vocab (Train):       {vocab_size_train:,} words")
    print(f"Train-Test Text Overlap:    {len(train_test_overlap):,} clauses ({leakage_stats['cross_split_leakage']['train_test_leakage_pct_of_test']}%)")
    print(f"Saved stats to:             {stats_path}")
    print("Figures saved in:           reports/figures/")
    print("=" * 60)
    return eda_stats


if __name__ == "__main__":
    analyze_dataset()
