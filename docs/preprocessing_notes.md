# FinDoc Preprocessing Architecture and Ablation Study

## 1. Pipeline Overview and Design Rationale
In legal and financial text processing, raw text presents two opposing challenges:
1. **Spurious Variations**: Specific contract party names (*"ABC Operating LLC"*, *"Borrower"*), dollar denominations (*"$15,400,000"*), execution dates (*"March 15, 2024"*), and section cross-references (*"Section 14.2(b)"*) create a combinatorial explosion in vocabulary without providing clause-type semantics.
2. **Deontic and Structural Modality**: Words frequently discarded in traditional NLP as stopwords—such as *shall*, *may*, *not*, *unless*, *without*, *pursuant*, *herein*—encode the core legal obligations, conditions, and liability thresholds that differentiate clause types (e.g. *Waivers* vs *Indemnifications* vs *Governing Laws*).

To evaluate these phenomena rigorously, we developed two configurable pipelines in `src/preprocess.py`:
- **Traditional Pipeline**: Standard lowercasing, whitespace tokenization, NLTK/spaCy stopword filtering, and lemmatization.
- **Specialized Legal Pipeline**:
  - Clause and section numbering stripping (e.g., `Section 12.01`, `(iv)`, `1.1.` $\to$ `[REF]`).
  - Typed entity masking: Currency amounts $\to$ `[MONEY]`, Dates $\to$ `[DATE]`, Durations $\to$ `[DURATION]`.
  - Defined term and entity normalization: Quoted parties $\to$ `[PARTY]`.
  - Boilerplate formatting collapse (removing underscore lines, repetitive dashes, multi-spaces).
  - Deduplication policy: Preserving only unique normalized clauses within the training set.

---

## 2. Real LEDGAR Before / After Examples

### Example 1: Waivers
- **Before**: `Except as otherwise set forth in this Debenture, the Company, for itself and its legal representatives, successors and assigns, expressly waives presentment, protest, demand, notice of dishonor, notice of nonpayment, not...`
- **After**: `except as otherwise set forth in this debenture, the [party], for itself and its legal representatives, successors and assigns, expressly waives presentment, protest, demand, notice of dishonor, notice of nonpayment, not...`
- **Effect**: Eliminates specific corporate identity; preserves crucial legal terms *waives*, *presentment*, *protest*, *demand*.

### Example 2: Governing Laws
- **Before**: `THIS AMENDMENT SHALL BE GOVERNED BY, AND INTERPRETED IN ACCORDANCE WITH, THE LAW OF THE STATE OF NEW YORK . The other provisions of Article IX of the Credit Agreement shall apply to this Amendment to the same extent as i...`
- **After**: `this amendment shall be governed by, and interpreted in accordance with, the law of the state of new york . the other provisions of article ix of the credit agreement shall apply to this amendment to the same extent as i...`
- **Effect**: Normalizes shouting uppercase boilerplate while maintaining legal jurisdiction entities.

### Example 3: Definitions / Cross-References
- **Before**: `Certain terms used in this Agreement are defined in this Section 1.1 . These terms, and the additional terms defined above, shall have the meanings assigned wherever the terms appear in this Agreement. These meanings ar...`
- **After**: `certain terms used in this agreement are defined in this [ref] . these terms, and the additional terms defined above, shall have the meanings assigned wherever the terms appear in this agreement. these meanings are also ...`
- **Effect**: Replaces arbitrary section numbering with standard token `[ref]`, eliminating unhelpful digit features.

### Example 4: Terms & Durations
- **Before**: `The " Term " of this Master Lease is the Initial Term plus all Renewal Terms, to the extent exercised. The initial term of this Master Lease (the " Initial Term ") shall commence on November 1, 2013 (the " Commencement ...`
- **After**: `the " term " of this master lease is the initial term plus all renewal terms, to the extent exercised. the initial term of this master lease (the " initial term ") shall commence on [date] (the " commencement date ") and...`
- **Effect**: Masks specific calendar dates with `[date]`, preventing over-fitting to historical filing years.

---

## 3. Preprocessing Ablation Study Results
We evaluated 6 configurations using a fixed model architecture (TF-IDF with 15,000 max features, unigram + bigram, sublinear TF scaling + Logistic Regression with $C=1.0$) trained on a standardized 20,000 sample of LEDGAR and evaluated on the full 10,000 validation split.

| Configuration | Vocab Size | Mean Length | Macro-F1 (Headline) | Micro-F1 | Accuracy | Total Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Raw Baseline** (Lowercase only) | 12,589 | 114.44 | 0.7074 | 0.8139 | 0.8139 | 33.7s |
| **2. Trad: Stopwords Removed** | 12,342 | 55.69 | 0.7072 | 0.8120 | 0.8120 | 199.2s |
| **3. Trad: Stopwords + Lemmatization** | 10,122 | 55.69 | 0.7085 | 0.8133 | 0.8133 | 195.9s |
| **4. Spec: Masking Only** | 12,507 | 114.12 | 0.7072 | 0.8137 | 0.8137 | 38.8s |
| **5. Spec: Full Legal Pipeline** | 12,402 | 112.61 | 0.7083 | 0.8142 | 0.8142 | 40.8s |
| **6. Spec: Full Legal + Deduplication** | **12,402** | **112.67** | **0.7086** | **0.8147** | **0.8147** | **37.0s** |

Visualizations:
- Vocabulary size comparison: [preprocessing_vocab_reduction.png](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/reports/figures/preprocessing_vocab_reduction.png)
- Clause length compression: [preprocessing_length_comparison.png](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/reports/figures/preprocessing_length_comparison.png)
- Macro-F1 ablation comparison: [preprocessing_ablation_f1.png](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/reports/figures/preprocessing_ablation_f1.png)

---

## 4. Key Engineering Insights: What Helped vs What Hurt

### 1. Stopword Removal Hurt Performance and Slowed Training
Filtering standard English stopwords decreased Micro-F1 from 0.8139 to 0.8120 and slightly reduced Macro-F1. Furthermore, running spaCy token-level stopword checks increased pipeline processing time by over 200x (0.8s $\to$ 177s). In legal contracts, relational stop words (e.g. *without*, *against*, *under*, *hereto*, *by*) carry high discriminative signal for rights and duties.

### 2. Lemmatization Offered Marginal Gain at Extreme Latency Cost
Lemmatization compressed the vocabulary by 19.6% (12,589 $\to$ 10,122) and slightly increased Macro-F1 to 0.7085 (+0.0011 over baseline). However, processing 20,000 clauses through the spaCy morphological analyzer took 173 seconds. On the full 60,000 corpus, this would add ~9 minutes per run with negligible practical difference.

### 3. Specialized Entity Masking and Normalization Maintained Strong Accuracy Fast
Masking dates, monetary amounts, and section numbering reduced noisy token variation while retaining pure execution speed (5.8 seconds vs 177 seconds for spaCy).

### 4. Deduplication Improved Generalization and Reduced Compute
Removing 245 exact duplicate clauses from the training subset produced the highest overall performance: **0.7086 Macro-F1** and **0.8147 Micro-F1**, while slightly lowering solver fit time. Eliminating redundant SEC boilerplate prevents the classifier from over-weighting repeated boilerplates.

---

## 5. Frozen Production Configuration
As formalized in [src/config.py](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/src/config.py):
- **Selected Pipeline**: `Spec: Full Legal + Deduplication`
- `lowercase = True`
- `remove_stopwords = False`
- `lemmatize = False`
- `remove_numbering = True`
- `mask_money = True`
- `mask_dates = True`
- `mask_durations = True`
- `normalize_parties = True`
- `collapse_boilerplate = True`
- `deduplicate = True`
