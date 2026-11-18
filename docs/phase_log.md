# FinDoc: Project Phase Log

## Phase 0: Setup and Feasibility

- **Date / Timestamp**: 2026-11-18
- **Hardware Environment**:
  - CPU: Intel Core i7-12650H (16 threads), 16 GB RAM
  - GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4.00 GB VRAM)
  - Driver Version: 610.91, CUDA UMD 13.3
  - PyTorch Version: `2.5.1+cu121` (`torch.cuda.is_available() == True`)
  - Initial Free VRAM: 3.23 GB / 4.00 GB
- **Software Stack**:
  - Python: 3.12.1 in `.venv`
  - Core libraries: `transformers==4.48.3`, `datasets==3.2.0`, `sentence-transformers==3.4.1`, `accelerate==1.2.1`, `scikit-learn==1.9.1`, `spacy==3.8.18` with `en_core_web_sm-3.8.0`, `rank_bm25==0.2.2`, `matplotlib==3.11.2`, `wordcloud==1.9.6`, `pandas==3.1.1`
  - Requirements pinned in `requirements.txt`.

### What Was Done
1. Initialized repository directory structure (`data/raw/`, `data/processed/`, `data/samples/`, `notebooks/`, `src/models/`, `results/`, `reports/figures/`, `docs/`, `tests/`).
2. Configured `.venv` and verified CUDA acceleration on the RTX 3050 Laptop GPU using PyTorch tensor operations.
3. Implemented robust dataset ingestion in `src/data.py`:
   - Primary source used: `coastalcph/lex_glue` (`ledgar` configuration).
   - Saved dataset to disk at `data/raw/ledgar`.
   - Verified offline loading with `HF_DATASETS_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` via unit test `tests/test_data_offline.py`.
   - Dataset properties verified:
     - Splits: `train` (60,000), `validation` (10,000), `test` (10,000). Total: 80,000 examples.
     - Classes: 100 legal clause categories.
     - Fields: `text` (string), `label` (int64 ClassLabel).
4. Conducted 200-step fp16 DistilBERT feasibility profiling benchmark (`src/benchmark_phase0.py`):
   - Benchmark configuration: `distilbert-base-uncased`, `batch_size=16`, `max_len=128`, `fp16=True`.
   - Peak VRAM allocated: 1.29 GB (Peak VRAM reserved: 1.45 GB) out of 4.00 GB total VRAM.
   - Safe VRAM headroom: 2.55 GB remaining on the RTX 3050 Laptop GPU.
   - Latency: 0.1007 seconds per step (9.93 steps/second).
   - Metrics saved to `results/phase0_benchmark.json`.

### Feasibility & Time Estimates
- Steps per epoch on train (60,000 examples, batch_size=16): 3,750 steps.
- Estimated training time for 1 epoch: ~6.30 minutes.
- Estimated training time for 3 epochs on full dataset: ~18.89 minutes (0.31 hours).
- **Flag status**: No planned experiment exceeds the 1-hour constraint on this GPU. Training on the full 60,000-example train split is feasible within 4 GB VRAM without OOM.

### What Was Decided
- Use `batch_size=16`, `max_len=128`, and `fp16=True` (with `torch.amp.autocast`) as the baseline configuration for transformer experiments to guarantee zero OOM risk and fast iteration (~19 min per 3-epoch run).
- Keep dataset cached locally at `data/raw/ledgar` and load offline for all subsequent phases.

### What Is Open
- Proceed to Phase 1: Dataset Exploration (EDA), token length distributions, class imbalance analysis (log scale), vocabulary statistics, n-grams, and representative class word clouds.

---

## Phase 1: Dataset Exploration (EDA)

- **Date / Timestamp**: 2026-11-18
- **Files Created**:
  - `src/eda.py`
  - `results/eda_stats.json`
  - `reports/figures/class_distribution_log.png`
  - `reports/figures/class_distribution_head_tail.png`
  - `reports/figures/token_length_distribution.png`
  - `reports/figures/token_length_by_class.png`
  - `reports/figures/top_ngrams.png`
  - `reports/figures/wordclouds_representative_classes.png`
  - `docs/eda_notes.md`
  - `notebooks/01_eda.py`

### What Was Done
1. Analyzed all 80,000 clauses across train (60,000), validation (10,000), and test (10,000) splits.
2. Verified 100 fine-grained contract clause classes with severe class imbalance:
   - Head: `Governing Laws` (3,167 train samples, 5.28%), `Notices` (2,493), `Counterparts` (2,427).
   - Tail: `Books` (23 train samples, 0.038%), `Exhibits` (27), `Cooperation` (34).
   - Imbalance ratio: 137.70x.
   - Pareto concentration: Top 10 classes account for 31.48% of the data; top 20 account for 48.33%.
3. Computed sequence length statistics per split and per category:
   - Train mean length: 114.96 tokens, median: 86 tokens.
   - 68.46% of clauses are $\le 128$ tokens; 91.84% are $\le 256$ tokens; 99.28% are $\le 512$ tokens.
4. Extracted training vocabulary metrics:
   - Total tokens: 6,897,339; unique vocabulary: 18,795 words; singletons: 5,513 (29.33%).
   - Extracted most frequent content unigrams, bigrams, and trigrams (e.g. `material adverse effect`, `transactions contemplated hereby`, `full force effect`).
   - Generated domain word clouds for 6 representative classes across head, mid, and tail tiers.
5. Quantified exact duplicates and cross-split leakage:
   - Within-train duplicates: 920 clauses (1.53%).
   - Cross-split leakage: 264 clauses shared between train and validation (2.65% of val); 265 clauses shared between train and test (2.67% of test).

### What Was Decided
- Primary metric: Macro-F1 must be our primary optimization and reporting metric due to the 137.7x class imbalance. Micro-F1 and accuracy will be tracked as secondary metrics.
- Max sequence length: Set default sequence length to 128 tokens for transformer training to preserve tight GPU memory bounds (< 1.5 GB VRAM) while covering ~68.5% of full clauses and the discriminative opening passage of all clauses.
- Data leakage awareness: Document the 2.67% test-set boilerplate overlap and design preprocessing/deduplication experiments in Phase 2.

### What Is Open
- Proceed to Phase 2: Preprocessing pipeline (traditional vs specialized legal normalizations, masking, ablation study with TF-IDF + Logistic Regression).

---

## Phase 2: Preprocessing Pipeline & Ablation

- **Date / Timestamp**: 2026-11-18
- **Files Created / Modified**:
  - `src/preprocess.py`
  - `src/generate_examples_phase2.py`
  - `src/ablation_phase2.py`
  - `src/config.py`
  - `results/preprocessing_ablation.json`
  - `reports/figures/preprocessing_vocab_reduction.png`
  - `reports/figures/preprocessing_length_comparison.png`
  - `reports/figures/preprocessing_ablation_f1.png`
  - `docs/preprocessing_notes.md`

### What Was Done
1. Designed and implemented a modular preprocessing engine in `src/preprocess.py` supporting toggleable traditional operations (lowercasing, stopword filtering, spaCy lemmatization) and specialized legal transformations (section/clause numbering removal, typed entity masking for `[MONEY]`, `[DATE]`, `[DURATION]`, defined party normalization `[PARTY]`, and SEC boilerplate collapse).
2. Generated 9 concrete before/after transformation examples from real LEDGAR provisions illustrating pattern substitution.
3. Conducted a controlled 6-configuration ablation study with TF-IDF + Logistic Regression on 20,000 training samples evaluated against the full 10,000 validation split:
   - Config 1 (Raw Baseline): Macro-F1 = 0.7074, Micro-F1 = 0.8139, Vocab = 12,589
   - Config 2 (Trad: Stopwords): Macro-F1 = 0.7072, Micro-F1 = 0.8120, Prep Time = 177.3s
   - Config 3 (Trad: Stopwords + Lemmatization): Macro-F1 = 0.7085, Micro-F1 = 0.8133, Vocab = 10,122, Prep Time = 173.4s
   - Config 4 (Spec: Masking Only): Macro-F1 = 0.7072, Micro-F1 = 0.8137, Prep Time = 5.8s
   - Config 5 (Spec: Full Legal): Macro-F1 = 0.7083, Micro-F1 = 0.8142, Prep Time = 7.8s
   - Config 6 (Spec: Full Legal + Deduplication): Macro-F1 = **0.7086**, Micro-F1 = **0.8147**, Accuracy = **0.8147**, Prep Time = 5.8s
4. Plotted comparative figures for vocabulary reduction, length distribution shifts, and Macro-F1 performance.
5. Froze the optimal configuration (`Spec: Full Legal + Deduplication`) in `src/config.py`.

### What Was Decided
- Keep stopwords and avoid full spaCy lemmatization in production training: Stopword removal harmed legal classification accuracy, while lemmatization brought a negligible +0.001 Macro-F1 gain at a 25x runtime penalty (~3 min vs 6 sec).
- Adopt full specialized regex normalization (`[REF]`, `[MONEY]`, `[DATE]`, `[DURATION]`, `[PARTY]`) combined with training set deduplication as the frozen standard for all downstream models in Phase 3, 4, and 7.

### What Is Open
- Proceed to Phase 3: Classical ML Baselines (Majority Class, TF-IDF + Logistic Regression, TF-IDF + Linear SVM, Complement Naive Bayes), tuning with a documented hyperparameter grid, evaluating on validation and test, and performing head-vs-tail class analysis.

---

## Phase 3: Classical ML Baselines & Head-vs-Tail Evaluation

- **Date / Timestamp**: 2026-11-18
- **Files Created / Modified**:
  - `src/features.py`
  - `src/models/baselines.py`
  - `results/baselines_results.json`
  - `reports/figures/baselines_performance_comparison.png`
  - `reports/figures/baselines_head_tail_f1.png`
  - `docs/baselines_notes.md`

### What Was Done
1. Implemented text feature extraction with TF-IDF vectorization in `src/features.py` using frozen specialized preprocessing parameters.
2. Implemented four classical baselines in `src/models/baselines.py`: Majority Class Dummy Classifier, Complement Naive Bayes, Linear Support Vector Machine (`LinearSVC`), and Multinomial Logistic Regression.
3. Conducted hyperparameter tuning on the validation split with documented validation grids:
   - ComplementNB: $\alpha=0.1$ selected ($\text{Val Macro-F1} = 0.6310$).
   - LinearSVC: $C=1.0$ selected ($\text{Val Macro-F1} = 0.8142$).
   - Logistic Regression: $C=5.0$ selected ($\text{Val Macro-F1} = 0.8007$).
4. Trained each model on the full 58,542 deduplicated training corpus and evaluated on the 10,000 validation split and 10,000 test split:
   - Majority Class: Test Macro-F1 = 0.0011, Micro-F1 = 0.0582
   - Complement Naive Bayes: Test Macro-F1 = 0.6238, Micro-F1 = 0.7719
   - Logistic Regression: Test Macro-F1 = 0.8164, Micro-F1 = 0.8732
   - Linear SVM: Test Macro-F1 = **0.8261**, Micro-F1 = **0.8790**, Accuracy = **0.8790**
5. Stratified performance into Head (Ranks 1–15), Mid (Ranks 16–50), and Tail (Ranks 51–100) tiers:
   - Linear SVM achieved Head F1 of 0.9169, Mid F1 of 0.8551, and Tail F1 of 0.7786.
   - ComplementNB suffered a steep 32.7 percentage point drop from Head (0.8385) to Tail (0.5113).

### What Was Decided
- Freeze Linear SVM ($C=1.0$) as the benchmark champion for classical NLP approaches (Test Macro-F1: 0.8261).
- Use the Head (0.9169) vs Tail (0.7786) disparity as the quantitative target for neural and transformer architectures in Phase 4.

### What Is Open
- Proceed to Phase 4: Neural and transformer models (fastText/EmbeddingBag, TextCNN/BiLSTM, Frozen SBERT/MiniLM + Logistic Regression, and DistilBERT fine-tuning with fp16, class imbalance ablation).

---

## Phase 4: Neural & Transformer Architectures

- **Date / Timestamp**: 2026-11-18
- **Files Created / Modified**:
  - `src/models/neural.py`
  - `src/models/sbert_model.py`
  - `src/models/transformer_train.py`
  - `src/models/imbalance_ablation.py`
  - `src/models/run_phase4_all.py`
  - `models/checkpoints/distilbert_ledgar/` (Saved best transformer weights, 268 MB)
  - `results/neural_results.json`
  - `results/imbalance_ablation.json`
  - `reports/figures/neural_models_comparison.png`
  - `reports/figures/neural_training_curves.png`
  - `reports/figures/imbalance_ablation_tail_f1.png`
  - `docs/neural_notes.md`

### What Was Done
1. Implemented and trained two PyTorch neural baselines in `src/models/neural.py`:
   - FastText-style EmbeddingBag (2.39M params): Test Macro-F1 = 0.7333, Micro-F1 = 0.8198, Latency = 0.015 ms/sample, VRAM = 61.9 MB, Train Time = 19.0s.
   - TextCNN (1D Multi-Filter, 2.61M params): Test Macro-F1 = 0.7543, Micro-F1 = 0.8353, Latency = 0.030 ms/sample, VRAM = 92.5 MB, Train Time = 32.1s.
2. Implemented frozen dense sentence embeddings in `src/models/sbert_model.py`:
   - Extracted 384-dimensional dense semantic features using `sentence-transformers/all-MiniLM-L6-v2` on GPU in FP16.
   - Trained Logistic Regression on dense embeddings: Test Macro-F1 = 0.7052, Micro-F1 = 0.8066, Latency = 0.527 ms/sample, Peak VRAM = 366.3 MB.
3. Fine-tuned `distilbert-base-uncased` (67.0M params) on the full LEDGAR train split across 3 epochs (10,977 steps) in `src/models/transformer_train.py`:
   - Epoch 1: Val Macro-F1 = 0.6969, Val Loss = 0.6625 (7.28 min)
   - Epoch 2: Val Macro-F1 = 0.7523, Val Loss = 0.5734 (7.36 min)
   - Epoch 3: Val Macro-F1 = 0.7725, Val Loss = 0.5373 (7.51 min)
   - Test Macro-F1 = **0.7710**, Micro-F1 = **0.8596**, Accuracy = **0.8596**.
   - Peak VRAM allocated: 1.29 GB (1.44 GB reserved) out of 4.00 GB.
   - Inference latency: 3.602 ms/sample (~278 samples/sec).
   - Saved full model checkpoint to `models/checkpoints/distilbert_ledgar/`.
4. Conducted a disciplined Imbalance Loss Ablation in `src/models/imbalance_ablation.py`:
   - Unweighted Cross-Entropy: Test Macro-F1 = 0.6384, Tail F1 = 0.5334.
   - Balanced Class Weights: Test Macro-F1 = **0.6685**, Tail F1 = **0.5968** (+6.34% improvement on low-resource legal provisions).
   - Focal Loss ($\gamma=2.0$): Test Macro-F1 = 0.6589, Tail F1 = 0.5651, Micro-F1 = 0.7758.

### What Was Decided
- Document and highlight the Occam's razor finding: Tuned Linear SVM (0.8261) and Logistic Regression (0.8164) outperform DistilBERT (0.7710) due to LEDGAR's exact legal boilerplate phrase structure and 128-token sequence truncation on transformers.
- Retain the fine-tuned DistilBERT checkpoint and frozen MiniLM embeddings for dense semantic retrieval in Phase 6 and hybrid CLI inference in Phase 7.
- Balanced class weights proved to be the single most effective method for mitigating long-tail class collapse (+6.34% Tail F1).

### What Is Open
- Proceed to Phase 5: Evaluation and Analysis (Grand comparison table, Top-20 confusion matrix, top confused class pairs, 25 hand-reviewed error examples with root-cause categorization, calibration checks, confidence-threshold coverage study).

---

## Phase 5: Evaluation & Analysis

- **Date / Timestamp**: 2026-11-18
- **Files Created / Modified**:
  - `src/evaluate.py`
  - `results/phase5_analysis.json`
  - `reports/figures/confusion_matrix_top20.png`
  - `reports/figures/top_confused_pairs.png`
  - `reports/figures/confidence_threshold_coverage.png`
  - `docs/analysis.md`

### What Was Done
1. Synthesized all experimental models into a **Grand Comparison Table** across parameters, memory footprint, training runtime, inference latency, and Macro/Micro/Head/Tail metrics.
   - Champion Model: Linear SVM ($C=1.0$) achieved **0.8261 Test Macro-F1**, 0.8790 Micro-F1, 0.9169 Head F1, 0.7786 Tail F1 with 0.004 ms/sample latency.
   - Neural Baselines: DistilBERT (0.7710), TextCNN (0.7543), FastText (0.7333), Frozen SBERT + LR (0.7052).
2. Computed and plotted the **Top-20 Class Confusion Matrix** and identified the **Top-15 Most Confused Pairs** (led by `Applicable Laws` $\to$ `Governing Laws`, `Modifications` $\to$ `Amendments`, `No Waivers` $\to$ `Waivers`, `Integration` $\to$ `Entire Agreements`).
3. Reviewed and categorized **25 hand-reviewed misclassifications** into an error taxonomy: Ambiguous/Overlapping Legal Taxonomy (36%), Multi-Intent Compound Provisions (28%), Class Imbalance/Tail Starvation (16%), Length Truncation/Buried Leads (12%), and Polarity Negation Inversion (8%).
4. Conducted a **Confidence-Threshold Abstention Study**:
   - At threshold $\tau=0.50$, system achieves **87.45% coverage** with accepted Macro-F1 improving to **0.8667** (+5.03%) and accuracy reaching **0.9245** (+5.13%).
   - The remaining 12.55% of clauses are designated as `"Uncertain"` for attorney review.
5. Penned an in-depth research analysis in `docs/analysis.md` explaining why tuned Linear SVM beat transformer fine-tuning (Occam's razor, formulaic SEC boilerplate, and transformer sequence length truncation).

### What Was Decided
- In the final production pipeline and CLI (Phase 7), implement the confidence threshold abstention ($\tau = 0.50$), labeling low-confidence predictions as `"Uncertain"`.
- Provide top-3 probability ranked predictions in the CLI to accommodate polysemous / compound provisions.

### What Is Open
- Proceed to Phase 6: Retrieval and Extraction (BM25 vs TF-IDF cosine vs SBERT dense embeddings retrieval with MRR / nDCG@10, and entity extraction rules for amounts, dates, durations, and organizations with manual precision audit).

---

## Phase 6: Retrieval & Key Entity Extraction

- **Date / Timestamp**: 2026-11-18
- **Files Created / Modified**:
  - `src/retrieval.py`
  - `src/extract.py`
  - `src/run_phase6.py`
  - `results/retrieval_extraction_results.json`
  - `reports/figures/retrieval_ranking_comparison.png`
  - `docs/retrieval_extraction_notes.md`

### What Was Done
1. Implemented a unified retrieval engine in `src/retrieval.py` supporting BM25 (Okapi), TF-IDF Cosine Similarity, and Dense Semantic SBERT (all-MiniLM-L6-v2) retrieval over 10,000 reference clauses.
2. Evaluated ranking metrics across 500 test queries with same-category match as the proxy relevance definition:
   - TF-IDF Cosine: Precision@5 = **0.7132**, MRR = **0.8237**, nDCG@10 = **0.7017**, Latency = 21.96 ms/query.
   - BM25: Precision@5 = 0.6780, MRR = 0.8121, nDCG@10 = 0.6664, Latency = 201.36 ms/query.
   - Dense SBERT: Precision@5 = 0.6692, MRR = 0.7873, nDCG@10 = 0.6543, Latency = **16.90 ms/query**.
3. Built an entity extraction engine in `src/extract.py` integrating typed regular expressions and spaCy NER for amounts, dates, durations, and parties.
4. Audited extraction quality over 100 actual test clauses:
   - Estimated Precisions: Amounts (96.5%), Dates (94.0%), Durations (92.5%), Parties (84.0%), Overall Precision: **91.8%**.
   - Documented 4 concrete failure patterns (relative date anchors, spelled-out currency sums, capitalized title ORG false positives).

### What Was Decided
- Select TF-IDF Cosine Similarity as the default similarity retrieval backend in the CLI pipeline due to its superior nDCG@10 (0.7017) and fast index build time (1.09s).
- Combine regex extraction with spaCy ORG filtering in the final production pipeline.

### What Is Open
- Proceed to Phase 7: Pipeline and CLI (`cli.py`, `pipeline.py`, contract segmenter, sample contracts, and comprehensive unit tests).

---

## Phase 7: Pipeline & CLI Tool

- **Date / Timestamp**: 2026-11-18
- **Files Created / Modified**:
  - `src/pipeline.py`
  - `src/cli.py`
  - `data/samples/sample_credit_agreement.txt`
  - `data/samples/sample_employment_agreement.txt`
  - `data/samples/sample_nda_agreement.txt`
  - `tests/test_preprocess.py`
  - `tests/test_segmentation.py`
  - `tests/test_pipeline.py`
  - `results/sample_credit_analysis.json`
  - `results/sample_employment_analysis.json`

### What Was Done
1. Implemented a robust rule-based contract clause segmenter and full `FinDocPipeline` in `src/pipeline.py` integrating classification, confidence thresholding ($\tau = 0.50$), reference retrieval, and typed entity extraction.
2. Built the production command-line interface in `src/cli.py` (`python -m src.cli analyze <path> [--threshold] [--output] [--no-retrieval]`) outputting an aligned terminal table plus structured JSON.
3. Created 3 realistic sample legal agreements in `data/samples/`: Credit and Revolving Loan Agreement, Executive Employment Agreement, and Mutual NDA.
4. Created and verified comprehensive unit tests:
   - `tests/test_preprocess.py` (Passed 5 assertions)
   - `tests/test_segmentation.py` (Passed 3 assertions)
   - `tests/test_pipeline.py` (Passed 2 end-to-end assertions)
5. Executed the CLI on the sample contracts:
   - `sample_credit_agreement.txt`: 7 clauses detected and accurately classified (Definitions, General, Interests, Representations, Indemnifications, Governing Laws, Notices) with 100% confidence, extracting $25,000,000, 36 months duration, and precedent matches.
   - `sample_employment_agreement.txt`: 6 clauses detected and classified (Positions, Base Salary, Terms, Non-Disparagement, Severability, Governing Laws) extracting $325,000, January 1, 2024, and notice durations.

### What Was Decided
- Retain modular decoupling between `pipeline.py` and `cli.py` so a FastAPI backend or Streamlit dashboard can import `FinDocPipeline` directly without code changes.

### What Is Open
- Proceed to Phase 8: Report and presentation support (`docs/report_outline.md`, literature review of 12 candidate papers, slide deck outline with figure mappings).

---

## Phase 8: Report & Presentation Support

- **Date / Timestamp**: 2026-11-18
- **Files Created / Modified**:
  - `docs/report_outline.md`
  - `docs/presentation_slides.md`
  - `README.md`

### What Was Done
1. Authored the comprehensive final report outline in `docs/report_outline.md` mapped strictly across the five required report sections:
   - Section 1: Introduction and Problem Formulation
   - Section 2: Dataset Characteristics & Exploratory Analysis (EDA)
   - Section 3: Preprocessing & Representation Engineering
   - Section 4: Modeling Architectures & Empirical Baselines
   - Section 5: Comprehensive Evaluation, Error Analysis & Real-World System
2. Conducted an academic literature review covering 12 foundational papers (LEDGAR, LexGLUE, CUAD, LEGAL-BERT, BERT, DistilBERT, fastText, TextCNN, Sentence-BERT, BM25, MiniLM, and Focal Loss), providing a 3-sentence summary and direct project relevance for each, with verification tags on specific bibliographic parameters.
3. Created a 5-content-slide presentation outline and defense script in `docs/presentation_slides.md` mapping every slide directly to the exact generated PNG figures in `reports/figures/` with complete speaker notes.
4. Updated `README.md` documenting the complete repository structure, reproduction steps, baseline leaderboard, and CLI commands.

### Project Final Status: COMPLETED & VERIFIED
All phases (Phase 0 through Phase 8) are fully implemented, empirically tested with actual outputs, saved to version-controlled JSON metrics and publication-quality figures, and thoroughly documented in the `docs/` directory.








