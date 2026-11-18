"""
Clause Retrieval Engine for FinDoc.
Implements:
1. BM25 (Okapi) Retrieval (rank_bm25)
2. TF-IDF Cosine Similarity Retrieval (scikit-learn)
3. Dense Semantic Retrieval (Sentence-Transformers all-MiniLM-L6-v2)

Evaluates on Precision@5, MRR, nDCG@10 using same-label category matching as proxy relevance.
"""

import time
import math
from typing import List, Dict, Tuple, Any

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.data import load_ledgar_offline
from src.preprocess import tokenize_simple, PreprocessingPipeline


class ClauseRetriever:
    """Unified retrieval interface for legal clauses."""
    def __init__(self, corpus_texts: List[str], corpus_labels: List[int], method: str = "bm25", device: str = "cpu"):
        self.corpus_texts = corpus_texts
        self.corpus_labels = corpus_labels
        self.method = method.lower()
        self.device = device

        print(f"Building {self.method.upper()} index over {len(corpus_texts):,} reference clauses...")
        t0 = time.perf_counter()

        if self.method == "bm25":
            self.tokenized_corpus = [tokenize_simple(t) for t in corpus_texts]
            self.index = BM25Okapi(self.tokenized_corpus)
        elif self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), sublinear_tf=True)
            self.corpus_matrix = self.vectorizer.fit_transform(corpus_texts)
        elif self.method in ["sbert", "dense"]:
            self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=self.device)
            if self.device == "cuda":
                self.encoder.half()
            self.corpus_embeddings = self.encoder.encode(corpus_texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")

        print(f"Index built in {time.perf_counter() - t0:.2f}s")

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Returns top_k most similar clauses for a query string."""
        if self.method == "bm25":
            tokens = tokenize_simple(query_text)
            scores = self.index.get_scores(tokens)
            top_indices = np.argsort(-scores)[:top_k]
            top_scores = scores[top_indices]
        elif self.method == "tfidf":
            q_vec = self.vectorizer.transform([query_text])
            sims = cosine_similarity(q_vec, self.corpus_matrix).flatten()
            top_indices = np.argsort(-sims)[:top_k]
            top_scores = sims[top_indices]
        elif self.method in ["sbert", "dense"]:
            q_emb = self.encoder.encode([query_text], show_progress_bar=False, normalize_embeddings=True)
            sims = np.dot(self.corpus_embeddings, q_emb.T).flatten()
            top_indices = np.argsort(-sims)[:top_k]
            top_scores = sims[top_indices]

        results = []
        for idx, score in zip(top_indices, top_scores):
            results.append({
                "corpus_index": int(idx),
                "text": self.corpus_texts[idx],
                "label": int(self.corpus_labels[idx]),
                "score": float(score)
            })
        return results


def compute_retrieval_metrics(
    retriever: ClauseRetriever,
    query_texts: List[str],
    query_labels: List[int],
    k: int = 10
) -> Dict[str, float]:
    """
    Evaluates retrieval performance using same-label category matching as proxy relevance.
    Computes: Precision@5, MRR, nDCG@10.
    """
    p_at_5_list = []
    mrr_list = []
    ndcg_list = []

    # Pre-compute IDCG@10 for binary relevance (if at least 1 relevant exists)
    # Discounts: 1 / log2(rank + 1)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    for q_text, q_label in zip(query_texts, query_labels):
        hits = retriever.query(q_text, top_k=k)
        relevances = [1 if h["label"] == q_label else 0 for h in hits]

        # 1. Precision@5
        p_at_5 = sum(relevances[:5]) / 5.0
        p_at_5_list.append(p_at_5)

        # 2. MRR (Mean Reciprocal Rank of first relevant item)
        mrr = 0.0
        for rank, rel in enumerate(relevances, start=1):
            if rel == 1:
                mrr = 1.0 / rank
                break
        mrr_list.append(mrr)

        # 3. nDCG@10
        dcg = np.sum(np.array(relevances) * discounts)
        ideal_num = min(sum(1 for l in retriever.corpus_labels if l == q_label), k)
        if ideal_num > 0:
            idcg = np.sum(discounts[:ideal_num])
            ndcg = dcg / idcg
        else:
            ndcg = 0.0
        ndcg_list.append(ndcg)

    return {
        "precision_at_5": round(float(np.mean(p_at_5_list)), 4),
        "mrr": round(float(np.mean(mrr_list)), 4),
        "ndcg_at_10": round(float(np.mean(ndcg_list)), 4),
        "query_count": len(query_texts)
    }
