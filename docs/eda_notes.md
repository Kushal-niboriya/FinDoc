# LEDGAR Dataset Exploration & Exploratory Data Analysis (EDA) Narrative

## 1. Dataset Overview & Provenance
The **LEDGAR** dataset (Tuggener et al., 2020; Chalkidis et al., 2022) is part of the **LexGLUE** legal NLP benchmark. Derived from material contracts filed with the **U.S. Securities and Exchange Commission (SEC) EDGAR system**, LEDGAR provides single-label classification of individual legal contract provisions into standardized clause types.

- **Total Corpus Size**: 80,000 legal clauses.
  - **Train split**: 60,000 clauses (75.0%)
  - **Validation split**: 10,000 clauses (12.5%)
  - **Test split**: 10,000 clauses (12.5%)
- **Target Variable**: 100 fine-grained contract clause categories (`ClassLabel` integer index mapping to legal clause labels).
- **Primary Schema**: `text` (string clause text), `label` (integer index in $[0, 99]$).

---

## 2. Class Imbalance and Long-Tail Distribution
The label distribution exhibits a steep **Zipfian / Pareto long-tail distribution**, representing a major challenge for standard classification metrics:

- **Top Class**: `Governing Laws` with 3,167 training instances (5.28% of the training set; 4,243 total instances).
- **Rarest Class**: `Books` with only 23 training instances (0.038% of the training set; 33 total instances).
- **Imbalance Ratio**: $\frac{\text{Count}(\text{Rank } 1)}{\text{Count}(\text{Rank } 100)} = \frac{3167}{23} \approx 137.70\times$.
- **Pareto Concentration**:
  - The top **10 categories** account for **31.48%** of the entire training corpus.
  - The top **20 categories** account for **48.33%** of the entire training corpus.
  - The bottom **50 tail categories** collectively represent just 18.2% of the dataset.

Visualizations:
- Log-scale distribution of all 100 categories: [class_distribution_log.png](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/reports/figures/class_distribution_log.png)
- Cumulative long-tail Pareto curve: [class_distribution_head_tail.png](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/reports/figures/class_distribution_head_tail.png)

### Key Takeaways for Modeling:
Standard accuracy and micro-averaged F1 will be heavily biased toward the head classes. **Macro-averaged F1** must serve as the primary headline metric to evaluate whether models successfully learn representations for low-resource tail categories.

---

## 3. Sequence Length Statistics & Transformer Truncation Analysis
Token lengths (computed via word-boundary tokenization) demonstrate that contract clauses vary considerably in length, with a right-skewed profile:

| Metric | Train Split | Validation Split | Test Split |
| :--- | :--- | :--- | :--- |
| **Total Count** | 60,000 | 10,000 | 10,000 |
| **Mean Length** | 114.96 tokens | 113.67 tokens | 108.80 tokens |
| **Median Length** | 86.0 tokens | 85.0 tokens | 78.0 tokens |
| **Std Dev** | 97.35 | 95.11 | 96.95 |
| **IQR (25% - 75%)** | 48 - 149 tokens | 47 - 150 tokens | 43 - 140 tokens |
| **95th Percentile** | 307 tokens | 299 tokens | 301 tokens |
| **Max Length** | 1,249 tokens | 914 tokens | 1,008 tokens |
| **% $\le 128$ tokens** | **68.46%** | **68.83%** | **71.70%** |
| **% $\le 256$ tokens** | **91.84%** | **92.07%** | **92.39%** |
| **% $\le 512$ tokens** | **99.28%** | **99.40%** | **99.26%** |

Visualizations:
- Token length distribution histogram: [token_length_distribution.png](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/reports/figures/token_length_distribution.png)
- Class-conditional length boxplots: [token_length_by_class.png](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/reports/figures/token_length_by_class.png)

### Architectural Implications:
A `max_seq_length` of 128 captures **68.5%** of all training clauses completely without any truncation. A sequence length of 256 captures **91.8%**. On the constrained 4 GB VRAM GPU, 128 tokens offers an optimal trade-off: it covers the discriminative opening and core provisions of virtually all clauses while keeping peak VRAM under 1.5 GB.

---

## 4. Vocabulary and Lexical Patterns
- **Total Training Tokens**: 6,897,339 words.
- **Unique Vocabulary**: 18,795 distinct words.
- **Singletons (hapax legomena)**: 5,513 terms (29.33% of the vocabulary appear exactly once), largely driven by idiosyncratic entity names, dollar amounts, and SEC formatting artifacts.
- **Top Domain Content Terms**:
  - High-frequency unigrams: `shall` (94,106), `agreement` (80,339), `company` (56,612), `party` (22,914), `borrower` (18,592), `executive` (18,572).
  - High-frequency bigrams: `agreement shall` (10,641), `set forth` (10,472), `administrative agent` (9,725), `transactions contemplated` (6,301), `shall deemed` (6,253).
  - High-frequency trigrams: `material adverse effect` (4,158), `transactions contemplated hereby` (3,311), `including without limitation` (3,303), `full force effect` (2,446).

Visualizations:
- Most frequent informative n-grams: [top_ngrams.png](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/reports/figures/top_ngrams.png)
- Word clouds for 6 representative classes: [wordclouds_representative_classes.png](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/reports/figures/wordclouds_representative_classes.png)
  - Head: *Governing Laws* (shows `state`, `new york`, `laws`, `construed`), *Amendments* (`writing`, `signed`, `consent`, `supplement`).
  - Mid: *Indemnifications* (`losses`, `claims`, `expenses`, `defend`), *Severability* (`invalid`, `illegal`, `unenforceable`, `provision`).
  - Tail: *Anti-Corruption Laws* (`fcpa`, `bribes`, `foreign`, `officials`), *Adjustments* (`stock`, `dividend`, `split`, `options`).

---

## 5. Duplicate Analysis and Cross-Split Data Leakage Audit
Contract drafting is inherently boilerplate-heavy: standard provisions (e.g. standard Delaware governing law or counterpart clauses) are replicated verbatim across many SEC filings.

### Within-Split Duplicates:
- **Train split duplicates**: 920 clauses (1.53% of train).
- **Validation split duplicates**: 32 clauses (0.32% of val).
- **Test split duplicates**: 64 clauses (0.64% of test).

### Cross-Split Overlap (Leakage):
- **Train $\cap$ Validation Overlap**: 264 distinct clause texts (2.65% of the validation split appear verbatim in the training set).
- **Train $\cap$ Test Overlap**: 265 distinct clause texts (2.67% of the test split appear verbatim in the training set).
- **Validation $\cap$ Test Overlap**: 65 distinct clause texts (0.65% of the test set).

### Scientific Finding & Caveat:
While 2.67% leakage is relatively modest in absolute numbers, in production and benchmark reporting this is critical: memorizing boilerplate could yield an artificial boost of up to ~2.7 percentage points in accuracy. In Phase 2 and subsequent evaluations, deduplication policies and out-of-distribution robustness checks must be explicitly analyzed.
