# FinDoc Classical ML Baselines and Head-vs-Tail Evaluation

## 1. Experimental Setup and Hyperparameter Tuning
In Phase 3, we implemented four classical baselines across the complete LEDGAR dataset (58,542 deduplicated training clauses, 10,000 validation clauses, 10,000 test clauses) using TF-IDF feature representations ($25,000$ max features, unigram + bigram, sublinear term-frequency scaling):

1. **Majority Class Baseline**: `DummyClassifier(strategy="most_frequent")`. Always predicts the most frequent class (*Governing Laws*). Serves as the minimal empirical floor.
2. **Complement Naive Bayes (`ComplementNB`)**: Specifically designed for severe class imbalance by estimating parameters from the complement of each class.
   - *Grid Search over validation set*: $\alpha \in [0.1, 0.5, 1.0]$ with length normalization `norm=True`.
   - *Selected*: $\alpha = 0.1$ ($\text{Val Macro-F1} = 0.6310$).
3. **Linear Support Vector Machine (`LinearSVC`)**: Maximizes margin between legal categories in the high-dimensional TF-IDF space.
   - *Grid Search over validation set*: $C \in [0.1, 0.5, 1.0]$.
   - *Selected*: $C = 1.0$ ($\text{Val Macro-F1} = 0.8142$).
4. **Multinomial Logistic Regression**: L2-regularized maximum-entropy model trained with the L-BFGS optimizer.
   - *Grid Search over validation set*: $C \in [0.5, 1.0, 5.0]$.
   - *Selected*: $C = 5.0$ ($\text{Val Macro-F1} = 0.8007$).

---

## 2. Comprehensive Baseline Results Table

| Model | Tuned Parameters | Val Macro-F1 | Val Micro-F1 | Test Macro-F1 (Headline) | Test Micro-F1 | Test Accuracy | Fit Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class** | `most_frequent` | 0.0009 | 0.0494 | **0.0011** | 0.0582 | 0.0582 | 0.01s |
| **Complement Naive Bayes** | $\alpha=0.1, \text{norm=True}$ | 0.6247 | 0.7699 | **0.6238** | 0.7719 | 0.7719 | 0.61s |
| **Logistic Regression** | $C=5.0, \text{solver=lbfgs}$ | 0.8007 | 0.8722 | **0.8164** | 0.8732 | 0.8732 | 154.3s |
| **Linear SVM (LinearSVC)** | $C=1.0, \text{max\_iter=2000}$ | 0.8142 | 0.8802 | **0.8261** | 0.8790 | 0.8790 | 68.5s |

Visualizations:
- Test Set Baseline Performance Comparison: [baselines_performance_comparison.png](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/reports/figures/baselines_performance_comparison.png)
- Head vs Mid vs Tail Macro-F1 Stratification: [baselines_head_tail_f1.png](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/reports/figures/baselines_head_tail_f1.png)

---

## 3. Head-vs-Tail Stratification Analysis
Due to the 137.7x class imbalance documented in Phase 1, global accuracy and Micro-F1 hide severe failure modes on rare categories. We stratified the 100 legal categories into three distinct frequency tiers based on their training volume:
- **Head Tier (Ranks 1–15)**: High-resource provisions (e.g. *Governing Laws*, *Notices*, *Counterparts*), representing $\approx 48\%$ of all data.
- **Mid Tier (Ranks 16–50)**: Moderate-resource provisions (e.g. *Severability*, *Indemnifications*, *Assignments*), representing $\approx 34\%$ of all data.
- **Tail Tier (Ranks 51–100)**: Low-resource provisions (e.g. *Anti-Corruption Laws*, *Adjustments*, *Books*), representing $\approx 18\%$ of all data.

### Tier Performance Comparison (Test Split):

| Model | Head F1 (Ranks 1–15) | Mid F1 (Ranks 16–50) | Tail F1 (Ranks 51–100) | Head-to-Tail Drop ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: |
| **Majority Class** | 0.0076 | 0.0000 | 0.0000 | -100% |
| **Complement Naive Bayes** | 0.8385 | 0.6923 | 0.5113 | **-32.72%** |
| **Logistic Regression** | 0.9117 | 0.8494 | 0.7646 | **-14.71%** |
| **Linear SVM** | **0.9169** | **0.8551** | **0.7786** | **-13.83%** |

### Critical Takeaways:
1. **Linear SVM is the Superior Classical Baseline**: With a Test Macro-F1 of **0.8261** and Accuracy of **0.8790**, Linear SVM outperforms Logistic Regression across every single tier. Margin maximization in sparse TF-IDF space provides natural resistance to high-dimensional colinear legal phrasing.
2. **Complement Naive Bayes Degrades Sharply on the Tail**: Despite the theoretical motivation of ComplementNB for class-imbalanced datasets, its independence assumption falters on complex legal language where multi-word compound phrases dictate category semantics. Its tail F1 drops to 0.5113, failing on 14 rare classes entirely.
3. **The Tail Challenge**: Even for Linear SVM, there is a **13.8 percentage point performance drop** between Head provisions (0.9169) and Tail provisions (0.7786). This sets a concrete challenge for Phase 4 neural and transformer models: neural encoders must improve semantic representations for rare tail categories.
