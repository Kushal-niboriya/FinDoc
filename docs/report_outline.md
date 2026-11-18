# FinDoc: Final University B.Tech Project Report Outline

**Project Title**: FinDoc: Clause Intelligence for Financial and Legal Contracts  
**Target Benchmark**: LexGLUE LEDGAR (SEC EDGAR Material Contracts, 80,000 Provisions across 100 Legal Classes)  
**Hardware Profile**: Intel Core i7-12650H, 16 GB RAM, NVIDIA GeForce RTX 3050 Laptop GPU (4.00 GB VRAM, CUDA 12.1)

---

## SECTION 1: Introduction and Problem Formulation
- **1.1 The Legal NLP Challenge**:
  - The exponential growth of commercial contracts in banking, corporate finance, and compliance.
  - Manual contract review is high-cost, error-prone, and bottlenecked by repetitive boilerplate examination.
- **1.2 Project Objectives**:
  1. *Rule-Based Clause Segmentation*: Decomposing unstructured contracts into discrete provisions.
  2. *Fine-Grained Clause Classification*: Accurately assigning one of 100 legal categories.
  3. *Reference Precedent Retrieval*: Locating semantically relevant clauses from indexed corporate filings.
  4. *Key Entity Extraction*: Extracting monetary values, dates, durations, and contracting parties.
  5. *Selective Classification (Abstention)*: Detecting uncertain edge cases ($\tau < 0.50$) to route to legal professionals.
- **1.3 Practical & Hardware Constraints**:
  - Zero paid API dependence; local offline inference within 4.0 GB VRAM.
  - Ethical AI and liability mitigation: Calibrated uncertainty prevents hallucinated legal advice.

---

## SECTION 2: Dataset Characteristics & Exploratory Analysis (EDA)
- **2.1 LEDGAR Corpus Provenance**:
  - Originates from SEC EDGAR 10-K, 10-Q, and 8-K filings.
  - Standardized benchmark: 60,000 train, 10,000 validation, 10,000 test provisions.
- **2.2 Extreme Class Imbalance (Pareto Distribution)**:
  - Top category (`Governing Laws`): 3,167 instances; rarest category (`Books`): 23 instances ($137.70\times$ imbalance ratio).
  - Top 10 classes constitute 31.48% of the data; top 20 constitute 48.33%.
  - *Figure reference*: `reports/figures/class_distribution_log.png` and `reports/figures/class_distribution_head_tail.png`.
- **2.3 Sequence Length Profile & Truncation Thresholds**:
  - Mean: 114.96 tokens, Median: 86 tokens, 95th percentile: 307 tokens.
  - 68.46% of clauses fit within $\le 128$ tokens; 91.84% fit within $\le 256$ tokens.
  - Hardware trade-off: `max_len=128` selected to maintain 1.29 GB peak VRAM.
  - *Figure reference*: `reports/figures/token_length_distribution.png` and `reports/figures/token_length_by_class.png`.
- **2.4 Lexical Patterns & Cross-Split Boilerplate Leakage**:
  - Unique vocabulary: 18,795 terms; singletons: 29.33%.
  - Documented 2.67% exact clause leakage between train and test splits due to verbatim corporate boilerplate.
  - *Figure reference*: `reports/figures/top_ngrams.png` and `reports/figures/wordclouds_representative_classes.png`.

---

## SECTION 3: Preprocessing & Representation Engineering
- **3.1 Traditional vs Specialized Legal NLP Pipeline**:
  - Why standard stopword filtering hurts legal classification: function words encode deontic modality (*shall*, *without*, *against*).
  - Lemmatization trade-off: 19.6% vocab reduction at a 25x runtime cost (173s vs 6s).
- **3.2 Specialized Pattern Normalization**:
  - Clause numbering & cross-reference collapse (`Section 4.1(a)` $\to$ `[REF]`).
  - Typed entity masking: Currency $\to$ `[MONEY]`, Dates $\to$ `[DATE]`, Durations $\to$ `[DURATION]`.
  - Defined corporate terms $\to$ `[PARTY]`.
- **3.3 Deduplication Policy**:
  - Removing 1,458 redundant training provisions improved Macro-F1 and accelerated training.
  - *Figure reference*: `reports/figures/preprocessing_vocab_reduction.png`, `preprocessing_length_comparison.png`, and `preprocessing_ablation_f1.png`.

---

## SECTION 4: Modeling Architectures & Empirical Baselines
- **4.1 Classical Models (CPU)**:
  - Majority Class baseline (Macro-F1: 0.0011).
  - Complement Naive Bayes ($\alpha=0.1$, Macro-F1: 0.6238).
  - Logistic Regression (L2, $C=5.0$, Macro-F1: 0.8164).
  - Linear Support Vector Machine ($C=1.0$, Macro-F1: **0.8261**).
  - *Figure reference*: `reports/figures/baselines_performance_comparison.png` and `baselines_head_tail_f1.png`.
- **4.2 Neural & Transformer Architectures (GPU Mixed Precision)**:
  - FastText EmbeddingBag (2.39M params, Macro-F1: 0.7333, 0.015 ms/sample).
  - Multi-Filter TextCNN (2.61M params, Macro-F1: 0.7543, 0.030 ms/sample).
  - Frozen SBERT (`all-MiniLM-L6-v2`) + Logistic Regression (Macro-F1: 0.7052).
  - Fine-Tuned DistilBERT (67.0M params, 3 epochs, fp16, Macro-F1: **0.7710**, 3.602 ms/sample).
  - *Figure reference*: `reports/figures/neural_models_comparison.png` and `neural_training_curves.png`.
- **4.3 Imbalance Mitigation Ablation**:
  - Unweighted Cross-Entropy (Tail F1: 0.5334).
  - Balanced Class Weights (Tail F1: **0.5968**, +6.34% improvement).
  - Focal Loss $\gamma=2.0$ (Tail F1: 0.5651, Micro-F1: 0.7758).
  - *Figure reference*: `reports/figures/imbalance_ablation_tail_f1.png`.

---

## SECTION 5: Comprehensive Evaluation, Error Analysis & Real-World System
- **5.1 Grand Performance Synthesis & The Occam's Razor Principle**:
  - Why Linear SVM outperformed fine-tuned DistilBERT (formulaic keyword n-grams vs 128-token sequence truncation).
- **5.2 Top-20 Confusion Matrix & Systematic Misclassification Pairs**:
  - Close semantic pairs: *Applicable Laws* vs *Governing Laws*, *Modifications* vs *Amendments*.
  - *Figure reference*: `reports/figures/confusion_matrix_top20.png` and `top_confused_pairs.png`.
- **5.3 25 Hand-Reviewed Error Classifications (Taxonomy)**:
  - Categorization into overlapping taxonomy, multi-intent clauses, tail starvation, and buried leads.
- **5.4 Model Calibration & Confidence Thresholding**:
  - Operating at $\tau = 0.50$ achieves **87.45% coverage** with accepted Macro-F1 of **0.8667** (+5.03%) and accuracy of **0.9245** (+5.13%).
  - *Figure reference*: `reports/figures/confidence_threshold_coverage.png`.
- **5.5 Precedent Retrieval & Entity Extraction Evaluation**:
  - TF-IDF Cosine achieves highest Precision@5 (0.7132) and nDCG@10 (0.7017).
  - Entity extraction achieves **91.8% overall precision** across amounts, dates, durations, and parties.
  - *Figure reference*: `reports/figures/retrieval_ranking_comparison.png`.
- **5.6 Real-World CLI System Demonstration**:
  - Demonstration on sample credit and employment agreements.

---

## LITERATURE REVIEW: 12 FOUNDATIONAL PAPERS

### 1. LEDGAR Dataset
- **Citation**: Tuggener, D., von Däniken, P., Peetz, T., & Cieliebak, M. (2020). *LEDGAR: A Large-Scale Multi-label Corpus for Text Classification of Legal Provisions in Contracts*. In Proceedings of the 12th Language Resources and Evaluation Conference (LREC 2020), pp. 1228–1234.
- **Summary**: Introduces the LEDGAR corpus extracted from US SEC EDGAR corporate filings, framing clause classification across standardized legal provisions. Demonstrates that legal provisions follow rigid structural and stylistic conventions that present unique challenges for NLP. Analyzes baseline document classification architectures on fine-grained contract provisions.
- **Relevance**: LEDGAR constitutes the core dataset for this project, defining the 100 legal clause categories, text boundaries, and primary evaluation splits.

### 2. LexGLUE Benchmark
- **Citation**: Chalkidis, I., Jana, A., Hartung, D., Bommarito, M., Androutsopoulos, I., Katz, D. M., & Aletras, N. (2022). *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English*. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL 2022), pp. 4310–4330.
- **Summary**: Establishes a standardized multi-task evaluation benchmark for English legal NLP modeled after GLUE and SuperGLUE. Integrates LEDGAR as a primary legal classification task alongside statutory interpretation, European court judgments, and US Supreme Court cases. Evaluates standard pre-trained language models and legal domain-specific variants.
- **Relevance**: Provides the standardized 60k/10k/10k partition of LEDGAR used in this work, establishing Macro-F1 and Micro-F1 as the standardized evaluation metrics.

### 3. CUAD (Contract Understanding Atticus Dataset)
- **Citation**: Hendrycks, D., Burns, C., Chen, A. S., & Ball, S. (2021). *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review*. In Thirty-fifth Conference on Neural Information Processing Systems (NeurIPS 2021) Datasets and Benchmarks Track.
- **Summary**: Curates 510 commercial legal contracts containing over 13,000 expert annotations mapped across 41 actionable contract review categories. Focuses on extractive question answering and clause highlighting for corporate due diligence. Demonstrates that general transformer models underperform without legal specialization.
- **Relevance**: Serves as our reference comparison corpus for commercial transactional agreements and guides our entity extraction and multi-intent clause taxonomy.

### 4. LEGAL-BERT
- **Citation**: Chalkidis, I., Fergadiotis, M., Malakasiotis, P., Aletras, N., & Androutsopoulos, I. (2020). *LEGAL-BERT: The Muppets straight out of Law School*. In Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 2898–2904.
- **Summary**: Pre-trains BERT architectures from scratch and performs domain adaptation on English legal text from legislation, court cases, and corporate contracts. Demonstrates substantial performance gains over general BERT on legal text classification and information extraction. Analyzes tokenizer fertility and domain vocabulary alignment for legal terminology.
- **Relevance**: Motivates domain-specific vocabulary and tokenization considerations in our preprocessing pipeline and explains tokenizer fertility challenges on legal contracts.

### 5. BERT (Bidirectional Encoder Representations from Transformers)
- **Citation**: Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. In Proceedings of NAACL-HLT 2019, pp. 4171–4186.
- **Summary**: Introduces bidirectional masked language modeling and next sentence prediction for deep contextual representation learning. Demonstrates state-of-the-art results across a broad spectrum of NLP tasks via fine-tuning. Popularized the transfer learning paradigm across computational linguistics.
- **Relevance**: Represents the parent architecture for transformer sequence classification heads evaluated in Phase 4.

### 6. DistilBERT
- **Citation**: Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. In NeurIPS EMC^2 Workshop 2019. [VERIFY exact workshop title and venue metadata].
- **Summary**: Applies knowledge distillation during pre-training to compress BERT by 40% in size while retaining 97% of its language understanding capabilities. Operates 60% faster during inference by removing token-type embeddings and reducing layer depth from 12 to 6. Demonstrates that compact transformers can run on resource-constrained hardware.
- **Relevance**: DistilBERT was fine-tuned as our primary transformer model in Phase 4, fitting safely within our 4 GB GPU VRAM constraint with 1.29 GB peak allocation.

### 7. FastText
- **Citation**: Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2017). *Bag of Tricks for Efficient Text Classification*. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2017), pp. 427–431.
- **Summary**: Proposes a simple linear model with rank constraint that averages word and subword n-gram embeddings followed by a linear classifier. Demonstrates that averaging word vectors achieves accuracy comparable to deep learning models while training orders of magnitude faster. Employs hierarchical softmax and hash-based token mapping for extreme efficiency.
- **Relevance**: Formed our `FastTextEmbeddingBag` baseline in Phase 4, training in just 19.0 seconds and running inference at 0.015 ms/sample.

### 8. TextCNN (Convolutional Neural Networks for Sentence Classification)
- **Citation**: Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification*. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP 2014), pp. 1746–1751.
- **Summary**: Applies 1D convolutions with variable filter widths (3, 4, 5) over word embedding matrices to capture multi-scale local phrase patterns. Employs max-over-time pooling to extract the most salient local features across variable sentence lengths. Demonstrates that shallow CNNs provide strong competitive performance on benchmark text classification tasks.
- **Relevance**: Implemented as our `TextCNN` architecture in Phase 4, achieving 0.7543 Macro-F1 and outperforming continuous bag-of-words by capturing local multi-word legal expressions.

### 9. Sentence-BERT (SBERT)
- **Citation**: Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP 2019), pp. 3982–3992.
- **Summary**: Modifies pre-trained BERT networks using siamese and triplet network structures to derive semantically meaningful sentence embeddings. Enables rapid semantic similarity comparison via cosine similarity, reducing semantic search latency from hours to milliseconds. Demonstrates superior performance on semantic textual similarity and information retrieval benchmarks.
- **Relevance**: Provided the dense embedding representations (`all-MiniLM-L6-v2`) for frozen linear probing in Phase 4 and dense precedent retrieval in Phase 6.

### 10. BM25 (Okapi BM25 Ranking Framework)
- **Citation**: Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval, 3(4), pp. 333–381. [VERIFY volume and page range].
- **Summary**: Details the theoretical underpinnings and empirical formulation of the Okapi BM25 ranking function for probabilistic information retrieval. Incorporates non-linear term frequency saturation ($k_1$) and document length normalization ($b$) to prevent score distortion on long documents. Remains the primary classical baseline for sparse information retrieval.
- **Relevance**: Evaluated in Phase 6 as our classical benchmark retrieval algorithm against TF-IDF and SBERT dense retrieval.

### 11. MiniLM
- **Citation**: Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*. In Advances in Neural Information Processing Systems (NeurIPS 2020), Vol. 33, pp. 5776–5788.
- **Summary**: Introduces self-attention relation distillation to compress large pre-trained transformers into compact models without restricting student architecture width. Distills both self-attention distributions and value-relation matrices from teacher models. Achieves competitive performance on language understanding while maintaining high throughput.
- **Relevance**: The `all-MiniLM-L6-v2` model (22.7M parameters) provided our dense vector representation, achieving 16.9 ms/query dense retrieval latency.

### 12. Focal Loss for Dense Object & Class-Imbalanced Detection
- **Citation**: Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal Loss for Dense Object Detection*. In Proceedings of the IEEE International Conference on Computer Vision (ICCV 2017), pp. 2980–2988.
- **Summary**: Addresses extreme foreground-background class imbalance by reshaping the standard cross-entropy loss function. Adds a dynamic modulating factor $(1 - p_t)^\gamma$ to down-weight easy well-classified examples and focus training on hard negative examples. Demonstrates that dynamically scaled loss functions overcome severe imbalance without destructive negative undersampling.
- **Relevance**: Evaluated in our Phase 4 imbalance ablation study ($\gamma=2.0$), boosting Tail class F1 performance by +3.17 percentage points without damaging Head class precision.
