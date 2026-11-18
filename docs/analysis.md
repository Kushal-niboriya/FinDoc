# FinDoc: Comprehensive Evaluation, Error Analysis, and System Dynamics

## 1. Grand Model Comparison Table
Across Phases 3 and 4, we implemented and empirically evaluated eight distinct model families on the full LEDGAR dataset (58,542 deduplicated training clauses, 10,000 validation clauses, 10,000 test clauses).

| Model Architecture | Model Family | Parameters | Peak VRAM | Training Time | Inference Latency | Test Macro-F1 (Headline) | Test Micro-F1 | Test Accuracy | Head F1 (1–15) | Mid F1 (16–50) | Tail F1 (51–100) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class** | Rule-Based | 0 | 0 MB | 0.01s | 0.001 ms | **0.0011** | 0.0582 | 0.0582 | 0.0073 | 0.0000 | 0.0000 |
| **Complement Naive Bayes** | Probabilistic | 2.50M | 0 MB | 0.61s | 0.003 ms | **0.6238** | 0.7701 | 0.7701 | 0.8341 | 0.6919 | 0.5131 |
| **FastText EmbeddingBag** | Neural Bag | 2.39M | 61.9 MB | 19.0s | 0.015 ms | **0.7333** | 0.8198 | 0.8198 | 0.8795 | 0.7710 | 0.6631 |
| **TextCNN (Multi-Filter)** | 1D ConvNet | 2.61M | 92.5 MB | 32.1s | 0.030 ms | **0.7543** | 0.8353 | 0.8353 | 0.8899 | 0.7981 | 0.6830 |
| **Frozen SBERT (MiniLM) + LR** | Dense Encoder | 22.7M | 366.3 MB | 62.7s | 0.527 ms | **0.7052** | 0.8066 | 0.8066 | 0.8671 | 0.7536 | 0.6228 |
| **DistilBERT (Fine-Tuned)** | Transformer | 67.0M | 1.29 GB | 23.03 min | 3.602 ms | **0.7710** | 0.8596 | 0.8596 | 0.9120 | 0.8298 | 0.6876 |
| **Logistic Regression** | Linear (L2) | 2.50M | 0 MB | 154.3s | 0.005 ms | **0.8164** | 0.8732 | 0.8732 | 0.9117 | 0.8494 | 0.7646 |
| **Linear SVM (Champion)** | Margin Linear | 2.50M | 0 MB | 68.5s | 0.004 ms | **0.8261** | **0.8790** | **0.8790** | **0.9169** | **0.8551** | **0.7786** |

Visualizations:
- Grand Baseline Comparison: [baselines_performance_comparison.png](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/reports/figures/baselines_performance_comparison.png)
- Neural Models Comparison: [neural_models_comparison.png](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/reports/figures/neural_models_comparison.png)

---

## 2. Top-20 Confusion Matrix & Top Confused Class Pairs
Analyzing the normalized confusion matrix across the top 20 classes reveals dense clusters of semantic proximity rather than random classification noise.

Visualization:
- Top-20 Confusion Matrix Heatmap: [confusion_matrix_top20.png](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/reports/figures/confusion_matrix_top20.png)
- Top 15 Misclassified Category Pairs: [top_confused_pairs.png](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/reports/figures/top_confused_pairs.png)

### The Top 10 Most Confused Category Pairs:
1. `Applicable Laws` $\to$ `Governing Laws` (35 errors): The legal distinction between which jurisdiction's substantive law applies (*Governing Law*) versus general statutory compliance obligations (*Applicable Laws*) shares almost identical lexical phrasing (*"laws of the State of..."*).
2. `Modifications` $\to$ `Amendments` (23 errors): Pure legal synonymy in corporate drafting; both describe altering the agreement terms (*"no modification or amendment shall be effective unless in writing"*).
3. `No Waivers` $\to$ `Waivers` (21 errors): Polarity inversion failure where standard legal clauses state that delay in exercising rights shall *not* operate as a waiver.
4. `Integration` $\to$ `Entire Agreements` (19 errors): Synonymous concepts (merger / integration clause stating that this contract supersedes all prior agreements).
5. `Successors` $\to$ `Binding Effects` (16 errors): Standard boilerplate (*"binding upon and inure to the benefit of successors and assigns"*).
6. `Tax Withholdings` $\to$ `Withholdings` (16 errors): Sub-category subsumption in employee and executive compensation.
7. `Defined Terms` $\to$ `Definitions` (16 errors) & `Definitions` $\to$ `Defined Terms` (14 errors): Bidirectional labeling noise across SEC filings.
8. `Authorizations` $\to$ `Authority` (14 errors): Synonymous corporate representation (*"duly authorized to enter this agreement"*).
9. `Indemnity` $\to$ `Indemnifications` (13 errors): Morphological noun inflection variants of identical provisions.
10. `Warranties` $\to$ `Survival` (13 errors): Contractual provisions stating that representations and warranties shall survive the closing.

---

## 3. Root-Cause Analysis: 25 Hand-Reviewed Misclassifications
A granular inspection of 25 actual misclassified test clauses revealed five distinct structural error mechanisms:

### Error Taxonomy Breakdown:
1. **Ambiguous / Overlapping Legal Taxonomy (36%)**:
   - *Example*: Test Sample #142 (`Applicable Laws` predicted as `Governing Laws` with 84.2% confidence).
   - *Text*: `"This agreement and all claims arising out of or related to this agreement shall be governed by, and construed in accordance with, the laws of the State of Delaware without regard to conflict of laws principles."`
   - *Root Cause*: The ground truth label in LEDGAR was tagged as `Applicable Laws`, but the provision is textually a canonical `Governing Law` clause. This represents annotation ambiguity in the SEC source data.
2. **Multi-Intent / Compound Provisions (28%)**:
   - *Example*: Test Sample #310 (`Survival` predicted as `Warranties` with 76.5% confidence).
   - *Text*: `"All representations and warranties made by the Borrower herein or in any other Loan Document shall survive the execution and delivery of this Agreement and the making of any Loans."`
   - *Root Cause*: The sentence contains two legal intents: asserting representations and declaring their survival. Single-label multi-class framing forces an arbitrary choice.
3. **Severe Class Imbalance / Tail Starvation (16%)**:
   - *Example*: Test Sample #914 (`Books` predicted as `Records` with 68.1% confidence).
   - *Text*: `"The Company shall keep proper books of record and account in which full, true and correct entries are made of all dealings and transactions in relation to its business and activities."`
   - *Root Cause*: `Books` has only 23 training examples, while `Records` has 184. The model biases toward the higher-frequency neighbor.
4. **Length Truncation & Buried Lead (12%)**:
   - *Example*: Test Sample #1205 (`Severability` predicted as `General` with 41.2% confidence).
   - *Text*: A 340-word clause whose opening 150 words recite boilerplate recital background before concluding in sentence 4: *"If any term of this Agreement is held invalid or unenforceable..."*
   - *Root Cause*: Models with fixed sequence cutoffs miss the dispositive operative language at the end of the clause.
5. **Polarity Negation Inversion (8%)**:
   - *Example*: Test Sample #421 (`No Waivers` predicted as `Waivers`).
   - *Root Cause*: The presence of terms *waiver*, *waive*, *waived* overpowers the single negative modifier *no failure or delay shall operate as a waiver*.

---

## 4. Model Calibration & Confidence Thresholding
In legal applications, deploying a model that outputs confident false assertions can incur substantial liability. A crucial production capability is **selective classification (abstention)**: predicting when confident and deferring to human attorneys when uncertain.

Visualization:
- Confidence Threshold Abstention Curves: [confidence_threshold_coverage.png](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/reports/figures/confidence_threshold_coverage.png)

### Key Empirical Findings:
- At baseline ($\tau = 0.0$, 100% coverage): Macro-F1 is **0.8164**, Accuracy is **0.8732**.
- At the recommended threshold ($\tau = 0.50$):
  - **Coverage**: **87.45%** of all contract provisions are classified automatically.
  - **Macro-F1**: Increases from 0.8164 to **0.8667** (+5.03 percentage points).
  - **Accuracy**: Increases from 0.8732 to **0.9245** (+5.13 percentage points).
  - **Abstention Rate**: 12.55% of clauses are routed to human review with the label `"Uncertain"`.
- At high-precision threshold ($\tau = 0.80$):
  - **Coverage**: **62.10%** of provisions.
  - **Accuracy**: **0.9577** (95.8% precision on accepted clauses).

---

## 5. Critical Technical Discussion: Why Classical Linear SVM Outperformed DistilBERT

One of the most consequential findings of this project is that **TF-IDF + Linear SVM (0.8261 Macro-F1)** and **Logistic Regression (0.8164)** consistently outperformed fine-tuned **DistilBERT (0.7710)** and frozen **MiniLM (0.7052)**.

### Architectural Explanations:
1. **Nature of Legal Draftsmanship (Formulaic Keyword Phrasing)**:
   Unlike conversational or literary text where syntactic fluidity dominates, SEC contracts are drafted using rigid, standardized boilerplate formulas. A clause is classified as *Governing Law* not because of deep hierarchical reasoning, but because of specific literal n-grams (*"governed by and construed in accordance with"*, *"laws of the State of"*). High-dimensional TF-IDF (25,000 dimensions) with sublinear term-frequency scaling directly maps these exact phrase coordinates.
2. **The 128-Token Truncation Penalty on Transformers**:
   Due to GPU VRAM constraints, DistilBERT was trained with `max_len=128`. In LEDGAR, **31.5% of training clauses exceed 128 tokens**. In clauses with lengthy introductory preamble, DistilBERT was truncated before the operative legal verb appeared, whereas TF-IDF processed the entire clause text regardless of length.
3. **Data Efficiency & Class Imbalance**:
   For rare tail classes with fewer than 50 examples, optimizing 67 million transformer parameters in 3 epochs leads to under-fitting or representation collapse. Linear SVM with an L2 margin maximizer creates clean separating hyperplanes in sparse vector space even with small sample counts.

---

## 6. Real-World Limitations
1. **Single-Label Simplification**: Real-world contract clauses often contain compound commitments (e.g. both an indemnity and a mandatory arbitration forum). Single-label classification artificially penalizes valid predictions.
2. **Corpus Distribution Shift**: LEDGAR reflects US SEC EDGAR public corporate filings. Performance may degrade on consumer contracts, NDAs, or Commonwealth legal jurisdictions (UK, India, Australia) without domain re-calibration.
3. **OCR and Formatting Noise**: Clean digital text is assumed; scanned PDFs with OCR transcription noise may require dedicated OCR cleanup pipelines before inference.
