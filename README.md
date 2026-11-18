# FinDoc: Clause Intelligence for Financial and Legal Contracts

FinDoc is an NLP and clause-intelligence engineering project developed for financial and legal contracts using the **LexGLUE LEDGAR benchmark** (80,000 corporate clauses across 100 legal categories).

FinDoc implements:
1. **Rule-Based Clause Segmentation**: Detects clause boundaries from raw contract text using hierarchical section and paragraph patterns.
2. **Multi-Class Clause Classification**: 100 fine-grained legal categories evaluated across 8 classical, neural, and transformer architectures.
3. **Precedent Retrieval Engine**: BM25, TF-IDF cosine, and dense SBERT retrieval against 10,000 reference SEC provisions.
4. **Key Entity Extraction**: Regex and spaCy NER extraction for monetary amounts, dates, durations, and contracting parties.
5. **Selective Classification (Abstention)**: Calibrated confidence thresholding ($\tau = 0.50$) routing uncertain clauses to legal professionals.

---

## Final Model Leaderboard (Test Split)

| Model Architecture | Model Family | Parameters | Peak VRAM | Test Macro-F1 (Headline) | Test Micro-F1 | Test Accuracy | Head F1 (1–15) | Tail F1 (51–100) | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM (Champion)** | Margin Linear | 2.50M | 0 MB | **0.8261** | **0.8790** | **0.8790** | **0.9169** | **0.7786** | 0.004 ms |
| **Logistic Regression** | Linear (L2) | 2.50M | 0 MB | **0.8164** | **0.8732** | **0.8732** | **0.9117** | **0.7646** | 0.005 ms |
| **DistilBERT (Fine-Tuned)** | Transformer | 67.0M | 1.29 GB | **0.7710** | **0.8596** | **0.8596** | **0.9120** | **0.6876** | 3.602 ms |
| **TextCNN (Multi-Filter)** | 1D ConvNet | 2.61M | 92.5 MB | **0.7543** | **0.8353** | **0.8353** | **0.8899** | **0.6830** | 0.030 ms |
| **FastText EmbeddingBag** | Neural Bag | 2.39M | 61.9 MB | **0.7333** | **0.8198** | **0.8198** | **0.8795** | **0.6631** | 0.015 ms |
| **Frozen SBERT (MiniLM) + LR** | Dense Encoder | 22.7M | 366.3 MB | **0.7052** | **0.8066** | **0.8066** | **0.8671** | **0.6228** | 0.527 ms |
| **Complement Naive Bayes** | Probabilistic | 2.50M | 0 MB | **0.6238** | **0.7701** | **0.7701** | **0.8341** | **0.5131** | 0.003 ms |
| **Majority Class** | Rule-Based | 0 | 0 MB | **0.0011** | **0.0582** | **0.0582** | **0.0073** | **0.0000** | 0.001 ms |

---

## Repository Structure
```
findoc/
  data/
    raw/ledgar/                  # Offline cached LEDGAR dataset
    processed/                   # Preprocessed matrices
    samples/                     # Sample contracts (Credit, Employment, NDA)
  docs/
    phase_log.md                 # Complete phase-by-phase engineering journal
    eda_notes.md                 # EDA findings, length and imbalance notes
    preprocessing_notes.md       # Preprocessing ablation analysis
    baselines_notes.md           # Classical ML baseline documentation
    neural_notes.md              # Neural & Transformer models & training curves
    analysis.md                  # Comprehensive error analysis & calibration study
    report_outline.md            # Final B.Tech report outline (5 sections + lit review)
    presentation_slides.md       # 5-slide presentation deck & defense script
  reports/
    figures/                     # Publication figures (distribution, confusion matrix, curves)
  results/                       # Version-controlled structured experiment JSONs
  src/
    data.py                      # Robust LEDGAR ingestion & offline loading
    eda.py                       # Exploratory analysis & plotting script
    preprocess.py                # Traditional & specialized legal preprocessing
    config.py                    # Frozen optimal configuration settings
    features.py                  # TF-IDF feature extraction
    models/
      baselines.py               # Classical baselines & tuning runner
      neural.py                  # FastText & TextCNN PyTorch architectures
      sbert_model.py             # Frozen SBERT dense embedding classifier
      transformer_train.py       # DistilBERT fp16 fine-tuning loop
      imbalance_ablation.py      # Class weighting & Focal Loss ablation
      run_phase4_all.py          # Phase 4 master experiment runner
    retrieval.py                 # BM25, TF-IDF cosine, and dense SBERT retrieval
    extract.py                   # Regex & spaCy NER entity extraction
    evaluate.py                  # Evaluation, confusion matrix & threshold study
    pipeline.py                  # End-to-end FinDocContractPipeline
    cli.py                       # Command-line interface
  tests/                         # Unit test suite
  requirements.txt               # Pinned dependencies
  README.md
```

---

## Quickstart & CLI Usage

### 1. Setup Virtual Environment
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run All Automated Unit Tests
```bash
python -m tests.test_preprocess
python -m tests.test_segmentation
python -m tests.test_pipeline
```

### 3. Analyze Contracts with the CLI
```bash
# Analyze credit agreement with table output & structured JSON export
python -m src.cli analyze data/samples/sample_credit_agreement.txt --output results/analysis_credit.json

# Analyze employment agreement with custom confidence threshold (0.60)
python -m src.cli analyze data/samples/sample_employment_agreement.txt --threshold 0.60
```
