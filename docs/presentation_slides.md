# FinDoc: B.Tech Presentation Slide Deck & Defense Script

## Slide Overview & Mapping Table

| Slide # | Slide Title | Core Message & Empirical Findings | Figures / Visuals to Embed |
| :---: | :--- | :--- | :--- |
| **Slide 1** | **Problem & System Architecture** | Automated clause segmentation, classification (100 classes), precedent retrieval, and key entity extraction under strict 4 GB VRAM constraints. | System Architecture Diagram / CLI Workflow Pipeline |
| **Slide 2** | **Corpus & Preprocessing** | LEDGAR Pareto long-tail (137.7x imbalance ratio); why traditional stopword removal hurts legal classification; typed entity masking and deduplication. | `reports/figures/class_distribution_log.png`<br>`reports/figures/preprocessing_ablation_f1.png` |
| **Slide 3** | **Model Comparison: The Occam's Razor Paradox** | Linear SVM (0.8261 Macro-F1) outperforms DistilBERT (0.7710) due to formulaic n-grams and 128-token sequence truncation on transformers. | `reports/figures/baselines_performance_comparison.png`<br>`reports/figures/neural_models_comparison.png` |
| **Slide 4** | **Error Taxonomy & Selective Classification** | Systematic errors (Applicable vs Governing Laws, synonymy, compound clauses); Confidence thresholding ($\tau=0.50$) achieves 92.5% accuracy with 87.5% coverage. | `reports/figures/confusion_matrix_top20.png`<br>`reports/figures/confidence_threshold_coverage.png` |
| **Slide 5** | **Live System Demo & Real-World Precedents** | End-to-end execution of `python -m src.cli analyze`: segmentation, top-3 ranked labels, entity extraction ($25M, 36 months), and TF-IDF precedent retrieval. | Terminal Output Table from `sample_credit_agreement.txt`<br>`reports/figures/retrieval_ranking_comparison.png` |

---

## Detailed Slide Content & Speaker Notes

### SLIDE 1: Problem Definition & FinDoc System Architecture
- **Header**: FinDoc: Clause Intelligence for Financial and Legal Contracts
- **Key Bullet Points**:
  - Commercial contracts in banking and corporate transactions run hundreds of pages; manual due diligence is expensive, repetitive, and vulnerable to fatigue.
  - FinDoc provides an automated, locally deployable NLP pipeline:
    1. **Rule-Based Segmentation**: Decomposing raw contracts into clean clause units.
    2. **Multi-Class Classification**: Mapping provisions into 100 standardized legal categories.
    3. **Precedent Retrieval**: Ranking similar clauses from 10,000 reference SEC filings.
    4. **Typed Entity Extraction**: Extracting amounts, calendar dates, durations, and parties.
  - **Constraint Fulfillment**: 100% offline, zero paid API wrappers, executed on an Intel i7-12650H and NVIDIA RTX 3050 Laptop GPU (4 GB VRAM) using PyTorch FP16.
- **Speaker Notes**:
  > "Good morning, respected committee members. Our project, FinDoc, tackles legal contract understanding using the LexGLUE LEDGAR benchmark. Commercial transactions are bottlenecked by manual review of hundreds of boilerplate clauses. FinDoc decomposes these contracts into provisions, classifies each against 100 categories, extracts financial entities, and retrieves precedent clauses from executed SEC filings—all running locally on an entry-level 4 GB laptop GPU without relying on paid APIs."

---

### SLIDE 2: Dataset Exploration & Preprocessing Innovations
- **Header**: Extreme Class Imbalance & Legal Preprocessing Ablation
- **Key Bullet Points**:
  - **The LEDGAR Long Tail**: 80,000 clauses across 100 categories. Severe Zipfian imbalance ($137.7\times$ ratio: `Governing Laws` has 3,167 train samples, while `Books` has only 23).
  - Top 10 classes account for 31.5% of the data; top 20 account for 48.3%.
  - **Preprocessing Findings**:
    - *Stopword filtering decreased performance* (-0.2% Micro-F1) and took 200x longer: legal function words (*shall*, *without*, *against*) carry critical deontic signal.
    - *Specialized typed masking* (`[MONEY]`, `[DATE]`, `[DURATION]`, `[PARTY]`) normalized spurious noise in 5.8 seconds.
    - *Deduplication* removed 1,458 duplicate boilerplate clauses, boosting Macro-F1 to **0.7086** and speeding up convergence.
- **Embedded Figures**:
  - Left: [class_distribution_log.png](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/reports/figures/class_distribution_log.png)
  - Right: [preprocessing_ablation_f1.png](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/reports/figures/preprocessing_ablation_f1.png)
- **Speaker Notes**:
  > "Our exploratory analysis revealed an extreme 137x class imbalance across the 100 categories, making Macro-F1 our non-negotiable primary metric. In our preprocessing ablation study, we proved that standard NLP stopword removal hurts legal classification because relational terms like 'under', 'pursuant to', and 'without' carry legal obligation meaning. Instead, our specialized entity masking and boilerplate deduplication delivered the highest Macro-F1 while running 25 times faster than spaCy lemmatization."

---

### SLIDE 3: Empirical Model Comparison: The Occam's Razor Paradox
- **Header**: Classical vs. Neural & Transformer Architectures
- **Key Bullet Points**:
  - Comprehensive evaluation across 8 distinct model families on 10,000 test clauses:
    - **Linear SVM (TF-IDF 25k)**: **0.8261 Macro-F1** | 0.8790 Micro-F1 | 0.004 ms/sample.
    - **Logistic Regression (L2)**: 0.8164 Macro-F1 | 0.8732 Micro-F1 | 0.005 ms/sample.
    - **DistilBERT (Fine-Tuned, FP16)**: 0.7710 Macro-F1 | 0.8596 Micro-F1 | 3.602 ms/sample.
    - **TextCNN (Multi-Filter 1D)**: 0.7543 Macro-F1 | 0.8353 Micro-F1 | 0.030 ms/sample.
    - **FastText (EmbeddingBag)**: 0.7333 Macro-F1 | 0.8198 Micro-F1 | 0.015 ms/sample.
    - **Frozen SBERT (MiniLM) + LR**: 0.7052 Macro-F1 | 0.8066 Micro-F1 | 0.527 ms/sample.
  - **The Engineering Paradox**: Why did Linear SVM outperform fine-tuned DistilBERT?
    1. *Boilerplate n-gram specificity*: SEC contracts rely on rigid phrase formulas that 25k TF-IDF features map directly.
    2. *Sequence Truncation*: Hardware constraints limited DistilBERT to 128 tokens, truncating 31.5% of long clauses, while TF-IDF indexed full-length clauses.
    3. *Data Efficiency on Rare Tail*: SVM margin maximization is more robust than 67M transformer weights on classes with < 50 examples.
- **Embedded Figures**:
  - Left: [baselines_performance_comparison.png](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/reports/figures/baselines_performance_comparison.png)
  - Right: [neural_models_comparison.png](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/reports/figures/neural_models_comparison.png)
- **Speaker Notes**:
  > "Here lies the most exciting empirical finding of our project: our tuned Linear SVM achieved an 0.8261 Macro-F1, cleanly beating our fine-tuned DistilBERT model at 0.7710. In computational linguistics, this illustrates Occam's razor: SEC legal provisions are written with formulaic n-grams that high-dimensional sparse linear classifiers capture perfectly with zero truncation penalty, whereas a 128-token transformer window truncates nearly a third of long clauses."

---

### SLIDE 4: Error Analysis, Calibration & Selective Abstention
- **Header**: Error Taxonomies & High-Precision Confidence Thresholding
- **Key Bullet Points**:
  - **Systematic Misclassification Clusters**:
    - Top confused pairs: *Applicable Laws* vs *Governing Laws* (35 errors), *Modifications* vs *Amendments* (23 errors), *No Waivers* vs *Waivers* (21 errors), *Integration* vs *Entire Agreements* (19 errors).
    - Root-cause breakdown: 36% Overlapping Taxonomy, 28% Multi-Intent Compound clauses, 16% Tail Starvation, 12% Length Truncation.
  - **Selective Classification (Abstention)**:
    - Setting threshold $\tau = 0.50$ flags uncertain predictions for human review.
    - **Coverage**: **87.45%** of provisions are classified automatically.
    - **Performance Boost**: Accepted Macro-F1 rises to **0.8667** (+5.03%) and Accuracy to **0.9245** (+5.13%).
- **Embedded Figures**:
  - Left: [confusion_matrix_top20.png](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/reports/figures/confusion_matrix_top20.png)
  - Right: [confidence_threshold_coverage.png](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/reports/figures/confidence_threshold_coverage.png)
- **Speaker Notes**:
  > "Because legal applications cannot tolerate unchecked errors, we conducted a rigorous 25-clause root-cause error review and built a confidence-threshold abstention mechanism. By deferring clauses where model confidence is below 50% to human attorneys, the system automatically processes 87.5% of the contract with an accepted accuracy of 92.5%, safely mitigating legal risk."

---

### SLIDE 5: Precedent Retrieval & Live CLI Demonstration
- **Header**: Precedent Discovery, Entity Extraction & Real-World CLI
- **Key Bullet Points**:
  - **Retrieval Performance**:
    - TF-IDF Cosine achieved **Precision@5: 0.7132** and **nDCG@10: 0.7017** with 21.9 ms latency, beating BM25 and dense SBERT.
  - **Entity Extraction Precision**:
    - **91.8% overall precision** across 100 audited clauses (Monetary Amounts: 96.5%, Dates: 94.0%, Durations: 92.5%, Parties: 84.0%).
  - **CLI Execution (`python -m src.cli analyze`)**:
    - Demonstrated on sample Credit Agreement (7 clauses detected, 100% confident, extracted $25M and 36 months, retrieved relevant precedent clauses in < 2 seconds).
    - Modular architecture prepared for FastAPI/Streamlit deployment.
- **Embedded Figures / Outputs**:
  - Left: [retrieval_ranking_comparison.png](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/reports/figures/retrieval_ranking_comparison.png)
  - Right: Terminal Table Output from `sample_credit_agreement.txt`.
- **Speaker Notes**:
  > "Finally, we tied everything together in our modular CLI tool. When fed a raw credit agreement, FinDoc segments the document, classifies all provisions with top-3 probabilities, extracts financial entities like dollar amounts and repayment durations, and surfaces the top 3 similar precedent clauses from our 10,000-clause SEC corpus. The entire system is fully modular and ready to power a FastAPI backend or web UI."
