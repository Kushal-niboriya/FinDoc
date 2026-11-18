"""
FinDoc End-to-End Contract Intelligence Pipeline.
1. Clause Segmentation: Rule-based boundary detection for raw legal agreements.
2. Clause Classification: Calibrated top-3 prediction with confidence threshold abstention.
3. Clause Retrieval: Top-3 similar reference clauses from LEDGAR reference corpus.
4. Entity Extraction: Extraction of amounts, dates, durations, and parties.
"""

import re
from typing import List, Dict, Any, Optional
import numpy as np

from src.data import load_ledgar_offline, get_dataset_metadata
from src.features import get_default_preprocessor, extract_tfidf_features
from src.preprocess import deduplicate_corpus
from src.retrieval import ClauseRetriever
from src.extract import extract_entities
from sklearn.linear_model import LogisticRegression

# Heading and boundary patterns
RE_SECTION_HEADER = re.compile(
    r"(?:^|\n\s*)(?:Section\s+\d+(?:[\.\-]\d+)*|Article\s+[IVXLCDM\d]+|\b\d+\.\s+[A-Z][a-zA-Z\s]{2,40}\.?)(?:\s*[\:\-\.]\s*|\n)",
    re.MULTILINE
)


def segment_contract(contract_text: str) -> List[Dict[str, Any]]:
    """
    Segments raw contract text into individual clauses using structural boundaries.
    Handles section headers, numbered paragraphs, and double newlines.
    """
    if not contract_text or not contract_text.strip():
        return []

    # Strategy 1: Look for explicit section headers
    matches = list(RE_SECTION_HEADER.finditer(contract_text))
    clauses = []

    if len(matches) >= 2:
        for i in range(len(matches)):
            start_idx = matches[i].start()
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(contract_text)
            raw_clause = contract_text[start_idx:end_idx].strip()
            
            # Extract header and body
            header_match = matches[i].group(0).strip()
            body = raw_clause[len(header_match):].strip()
            if not body:
                body = raw_clause

            if len(body) > 30:  # Filter out trivial blank headers
                clauses.append({
                    "clause_id": len(clauses) + 1,
                    "header": header_match,
                    "text": raw_clause
                })
    else:
        # Strategy 2: Split by double newlines or numbered paragraphs
        paragraphs = re.split(r"\n\s*\n+", contract_text.strip())
        for p in paragraphs:
            p_clean = p.strip()
            if len(p_clean) > 30:
                clauses.append({
                    "clause_id": len(clauses) + 1,
                    "header": p_clean[:40].replace("\n", " ") + "...",
                    "text": p_clean
                })

    # Fallback: Single clause if text is short
    if not clauses and contract_text.strip():
        clauses.append({
            "clause_id": 1,
            "header": "Clause 1",
            "text": contract_text.strip()
        })

    return clauses


class FinDocPipeline:
    """
    End-to-End Contract Intelligence Pipeline.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.50,
        corpus_size: int = 10000,
        enable_retrieval: bool = True
    ):
        self.confidence_threshold = confidence_threshold
        self.enable_retrieval = enable_retrieval

        print("[FinDocPipeline] Initializing models and reference indices...")
        ds = load_ledgar_offline()
        meta = get_dataset_metadata(ds)
        self.label_names = meta["label_names"]
        self.num_classes = len(self.label_names)

        # 1. Preprocessor
        self.preprocessor = get_default_preprocessor()

        # 2. Classifier (Calibrated Logistic Regression on 25k TF-IDF features)
        print("[FinDocPipeline] Fitting calibrated classifier...")
        train_texts = self.preprocessor.transform_batch(ds["train"]["text"])
        train_texts, train_labels, _ = deduplicate_corpus(train_texts, ds["train"]["label"])

        self.vectorizer, X_train, _, _ = extract_tfidf_features(
            train_texts, train_texts[:10], train_texts[:10],
            max_features=25000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.classifier = LogisticRegression(C=5.0, max_iter=200, random_state=42, solver="lbfgs")
        self.classifier.fit(X_train, train_labels)

        # 3. Retriever
        if self.enable_retrieval:
            print(f"[FinDocPipeline] Building reference retrieval index over {corpus_size:,} clauses...")
            corpus_subset_texts = ds["train"]["text"][:corpus_size]
            corpus_subset_labels = ds["train"]["label"][:corpus_size]
            self.retriever = ClauseRetriever(
                corpus_subset_texts,
                corpus_subset_labels,
                method="tfidf"
            )
        else:
            self.retriever = None

        print("[FinDocPipeline] Ready for analysis!")

    def analyze_clause(self, clause_text: str, top_k_similar: int = 3) -> Dict[str, Any]:
        """Analyzes a single contract clause."""
        processed_text = self.preprocessor.transform_text(clause_text)

        # Vectorize and predict
        vec = self.vectorizer.transform([processed_text])
        probs = self.classifier.predict_proba(vec)[0]

        # Top 3 predictions
        top_indices = np.argsort(-probs)[:3]
        top_predictions = [
            {
                "label": self.label_names[idx],
                "confidence": round(float(probs[idx]), 4),
                "confidence_pct": round(float(probs[idx] * 100), 2)
            }
            for idx in top_indices
        ]

        top_label = top_predictions[0]["label"]
        top_conf = top_predictions[0]["confidence"]
        is_uncertain = top_conf < self.confidence_threshold
        status = "Uncertain (Human Review Recommended)" if is_uncertain else "Confident"

        # Entity Extraction
        entities = extract_entities(clause_text)

        # Similar Reference Clauses
        similar_clauses = []
        if self.retriever:
            hits = self.retriever.query(clause_text, top_k=top_k_similar)
            for hit in hits:
                similar_clauses.append({
                    "score": round(hit["score"], 4),
                    "category": self.label_names[hit["label"]],
                    "text_snippet": hit["text"][:140] + ("..." if len(hit["text"]) > 140 else "")
                })

        return {
            "top_label": top_label,
            "confidence": top_conf,
            "status": status,
            "is_uncertain": is_uncertain,
            "top_3_predictions": top_predictions,
            "entities": entities,
            "similar_clauses": similar_clauses
        }

    def analyze_contract(self, contract_text: str, top_k_similar: int = 3) -> Dict[str, Any]:
        """Segments and analyzes full contract."""
        clauses = segment_contract(contract_text)
        analyzed_clauses = []

        uncertain_count = 0
        all_entities = {"amounts": [], "dates": [], "durations": [], "parties": []}

        for c in clauses:
            res = self.analyze_clause(c["text"], top_k_similar=top_k_similar)
            if res["is_uncertain"]:
                uncertain_count += 1

            for k in all_entities:
                all_entities[k].extend(res["entities"][k])

            analyzed_clauses.append({
                "clause_id": c["clause_id"],
                "header": c["header"],
                "text": c["text"],
                "analysis": res
            })

        # Deduplicate contract-wide entities
        for k in all_entities:
            all_entities[k] = list(dict.fromkeys(all_entities[k]))

        return {
            "total_clauses": len(clauses),
            "confident_clauses": len(clauses) - uncertain_count,
            "uncertain_clauses": uncertain_count,
            "confidence_threshold_used": self.confidence_threshold,
            "contract_entities": all_entities,
            "clauses": analyzed_clauses
        }
