"""
Phase 6: Retrieval & Extraction Benchmark Runner.
1. Evaluates BM25 vs TF-IDF vs Dense SBERT Retrieval:
   - Queries: 500 test clauses
   - Corpus: 10,000 training clauses
   - Metrics: Precision@5, MRR, nDCG@10 (using same-label as proxy relevance)
2. Audits Key Entity Extraction (Amounts, Dates, Durations, Parties) on 100 clauses.
3. Outputs:
   - results/retrieval_extraction_results.json
   - reports/figures/retrieval_ranking_comparison.png
"""

import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from src.data import load_ledgar_offline
from src.retrieval import ClauseRetriever, compute_retrieval_metrics
from src.extract import audit_extraction_rules

FIGURES_DIR = Path("reports/figures")
RESULTS_DIR = Path("results")
CORPUS_SIZE = 10000
QUERY_SIZE = 500


def run_phase6():
    print("=" * 80)
    print("PHASE 6: RETRIEVAL & EXTRACTION SYSTEM EVALUATION")
    print("=" * 80)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load dataset
    print("\n[1/4] Loading LEDGAR dataset for retrieval benchmark...")
    ds = load_ledgar_offline()
    
    # Subsample 10,000 corpus clauses and 500 test query clauses
    corpus_texts = ds["train"]["text"][:CORPUS_SIZE]
    corpus_labels = ds["train"]["label"][:CORPUS_SIZE]
    query_texts = ds["test"]["text"][:QUERY_SIZE]
    query_labels = ds["test"]["label"][:QUERY_SIZE]

    print(f"Corpus size: {len(corpus_texts):,} clauses | Query size: {len(query_texts):,} clauses")

    # 1. Retrieval Benchmarks
    retrieval_results = {}
    methods = ["bm25", "tfidf", "sbert"]

    print("\n[2/4] Benchmarking retrieval algorithms (Proxy relevance = Same Category)...")
    for method in methods:
        print(f"\n---> Evaluating {method.upper()} Retrieval...")
        t_build = time.perf_counter()
        retriever = ClauseRetriever(corpus_texts, corpus_labels, method=method, device=device)
        build_time = time.perf_counter() - t_build

        t_eval = time.perf_counter()
        metrics = compute_retrieval_metrics(retriever, query_texts, query_labels, k=10)
        eval_time = time.perf_counter() - t_eval
        latency_ms = (eval_time / len(query_texts)) * 1000

        metrics["index_build_sec"] = round(build_time, 2)
        metrics["query_latency_ms"] = round(latency_ms, 2)

        retrieval_results[method] = metrics
        print(f"  {method.upper()} Results -> P@5: {metrics['precision_at_5']:.4f} | MRR: {metrics['mrr']:.4f} | nDCG@10: {metrics['ndcg_at_10']:.4f} | Latency: {latency_ms:.2f} ms/query")

    # Plot Retrieval Comparison
    print("\n[3/4] Generating retrieval comparison figure...")
    plt.figure(figsize=(10, 5.5))
    x = np.arange(len(methods))
    width = 0.25

    p5_vals = [retrieval_results[m]["precision_at_5"] for m in methods]
    mrr_vals = [retrieval_results[m]["mrr"] for m in methods]
    ndcg_vals = [retrieval_results[m]["ndcg_at_10"] for m in methods]

    plt.bar(x - width, p5_vals, width, label="Precision@5", color="#1f77b4")
    plt.bar(x, mrr_vals, width, label="MRR (Mean Reciprocal Rank)", color="#2ca02c")
    plt.bar(x + width, ndcg_vals, width, label="nDCG@10", color="#ff7f0e")

    plt.xticks(x, ["BM25 (Okapi)", "TF-IDF (Cosine)", "Dense SBERT (MiniLM)"], fontsize=11, fontweight="bold")
    plt.ylabel("Score (0.0 to 1.0)", fontsize=11)
    plt.title("Contract Clause Retrieval Performance Comparison\n(Proxy Relevance: Same Category Match)", fontsize=12, fontweight="bold", pad=12)
    plt.ylim(0, 1.0)
    plt.legend(frameon=True, loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(len(methods)):
        plt.text(x[i] - width, p5_vals[i] + 0.02, f"{p5_vals[i]:.3f}", ha="center", fontsize=8.5, fontweight="bold")
        plt.text(x[i], mrr_vals[i] + 0.02, f"{mrr_vals[i]:.3f}", ha="center", fontsize=8.5, fontweight="bold")
        plt.text(x[i] + width, ndcg_vals[i] + 0.02, f"{ndcg_vals[i]:.3f}", ha="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "retrieval_ranking_comparison.png", dpi=300)
    plt.close()

    # 2. Entity Extraction Audit
    print("\n[4/4] Running Key Entity Extraction Audit over 100 clauses...")
    sample_100_clauses = ds["test"]["text"][:100]
    extraction_audit = audit_extraction_rules(sample_100_clauses)
    print(f"Extracted entities summary across 100 clauses: {extraction_audit['total_extracted_counts']}")
    print(f"Estimated overall precision: {extraction_audit['estimated_precision']['overall_precision']*100:.1f}%")

    output_payload = {
        "metadata": {
            "retrieval_corpus_size": CORPUS_SIZE,
            "retrieval_query_size": QUERY_SIZE,
            "proxy_relevance_definition": "Binary relevance: true clause category equals retrieved reference clause category. Category match serves as an objective semantic proxy.",
            "extraction_clauses_audited": 100
        },
        "retrieval_benchmarks": retrieval_results,
        "extraction_audit": extraction_audit
    }

    with open(RESULTS_DIR / "retrieval_extraction_results.json", "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\nPhase 6 successfully completed! Results saved to {RESULTS_DIR / 'retrieval_extraction_results.json'}")
    return output_payload


if __name__ == "__main__":
    run_phase6()
