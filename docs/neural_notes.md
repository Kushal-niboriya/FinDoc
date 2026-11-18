# FinDoc Neural and Transformer Model Suite (Phase 4)

## 1. Architectural Overview & Experimental Setup
In Phase 4, we implemented, optimized, and evaluated four neural and transformer architectures on the full LEDGAR dataset (58,542 deduplicated training clauses, 10,000 validation clauses, 10,000 test clauses) executed on an NVIDIA GeForce RTX 3050 Laptop GPU (4.00 GB VRAM) using PyTorch with CUDA 12.1 and mixed-precision (FP16):

1. **FastText-Style EmbeddingBag (`nn.EmbeddingBag`)**:
   - Continuous bag-of-words representation with global average pooling and linear classification head.
   - 128-dimensional dense word embeddings over a 18,570-word vocabulary.
   - Ultra-lightweight: 2,389,860 parameters (~9.1 MB).
2. **Multi-Filter TextCNN (Kim, 2014)**:
   - 1D Convolutional Neural Network with 3 parallel filter banks (kernel sizes 3, 4, 5, with 128 channels each).
   - Global 1D max-over-time pooling followed by Dropout (0.3) and a 100-class dense projection.
   - Captures local n-gram patterns and legal phrases: 2,612,452 parameters (~10.0 MB).
3. **Frozen Dense Sentence Embeddings (MiniLM) + Logistic Regression**:
   - `sentence-transformers/all-MiniLM-L6-v2` encoder (22,713,216 parameters) running in FP16 on GPU.
   - Generates 384-dimensional normalized semantic embeddings for all clauses.
   - Multinomial Logistic Regression ($C=2.0$, L-BFGS) trained on the dense representation.
4. **Fine-Tuned DistilBERT (`distilbert-base-uncased`)**:
   - 6-layer, 768-hidden, 12-head transformer encoder fine-tuned end-to-end with a 100-class sequence classification head (67,030,372 parameters).
   - Trained for 3 full epochs (10,977 optimization steps) with `max_len=128`, `batch_size=16`, `lr=3e-5`, linear warmup over 10% of steps, and PyTorch AMP FP16.

---

## 2. Comprehensive Model Comparison (Test Split)

| Model Architecture | Parameters | Peak VRAM | Training Time | Latency (ms/sample) | Test Macro-F1 (Headline) | Test Micro-F1 | Test Accuracy | Head F1 (1–15) | Mid F1 (16–50) | Tail F1 (51–100) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class** | 0 | 0 MB | 0.01s | 0.001 ms | **0.0011** | 0.0582 | 0.0582 | 0.0076 | 0.0000 | 0.0000 |
| **Complement Naive Bayes** | 2.5M | 0 MB | 0.61s | 0.003 ms | **0.6238** | 0.7719 | 0.7719 | 0.8385 | 0.6923 | 0.5113 |
| **FastText EmbeddingBag** | 2.39M | 61.9 MB | 19.0s | 0.015 ms | **0.7333** | 0.8198 | 0.8198 | 0.8795 | 0.7710 | 0.6631 |
| **TextCNN (Multi-Filter)** | 2.61M | 92.5 MB | 32.1s | 0.030 ms | **0.7543** | 0.8353 | 0.8353 | 0.8899 | 0.7981 | 0.6830 |
| **Frozen SBERT (MiniLM) + LR** | 22.7M | 366.3 MB | 62.7s | 0.527 ms | **0.7052** | 0.8066 | 0.8066 | 0.8671 | 0.7536 | 0.6228 |
| **DistilBERT (Fine-Tuned)** | 67.0M | 1.29 GB | 23.03 min | 3.602 ms | **0.7710** | 0.8596 | 0.8596 | 0.9120 | 0.8298 | 0.6876 |
| **Linear SVM (Classical)** | 2.5M | 0 MB | 68.5s | 0.004 ms | **0.8261** | 0.8790 | 0.8790 | 0.9169 | 0.8551 | 0.7786 |

Visualizations:
- Neural & Transformer Models Comparison: [neural_models_comparison.png](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/reports/figures/neural_models_comparison.png)
- DistilBERT Loss and Convergence Curves: [neural_training_curves.png](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/reports/figures/neural_training_curves.png)

---

## 3. DistilBERT Training Dynamics and Hardware Profile
- **Convergence Behavior**:
  - Epoch 1: Train Loss = 1.5951, Val Loss = 0.6625, Val Macro-F1 = 0.6969, Val Acc = 0.8202 (Time: 7.28 min)
  - Epoch 2: Train Loss = 0.5623, Val Loss = 0.5734, Val Macro-F1 = 0.7523, Val Acc = 0.8480 (Time: 7.36 min)
  - Epoch 3: Train Loss = 0.3968, Val Loss = 0.5373, Val Macro-F1 = 0.7725, Val Acc = 0.8596 (Time: 7.51 min)
- **VRAM Utilization**: Peak allocated VRAM was **1.29 GB** (peak reserved **1.44 GB**), leaving over 2.5 GB of safe headroom on the 4.00 GB laptop GPU.
- **Inference Throughput**: DistilBERT processed 10,000 test clauses in 36.0 seconds (3.60 ms per sample $\approx 278$ clauses/sec).
- **Artifact**: Best checkpoint saved to `models/checkpoints/distilbert_ledgar/` (268.1 MB).

---

## 4. Class Imbalance Mitigation Ablation Study
We conducted an empirical ablation isolating the effect of loss formulations on the long-tail distribution:

| Strategy | Test Macro-F1 | Test Micro-F1 | Head F1 (1–15) | Mid F1 (16–50) | Tail F1 (51–100) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Unweighted Cross-Entropy** | 0.6384 | **0.7710** | 0.8427 | 0.7008 | 0.5334 |
| **2. Balanced Class Weights** | **0.6685** | 0.7461 | 0.8312 | 0.7014 | **0.5968 (+6.34%)** |
| **3. Focal Loss ($\gamma=2.0$)** | 0.6589 | 0.7758 | **0.8495** | **0.7112** | 0.5651 (+3.17%) |
| **4. Focal Loss + Class Weights** | 0.6372 | 0.6691 | 0.7178 | 0.6947 | 0.5727 |

Visualization:
- Imbalance Loss Ablation Chart: [imbalance_ablation_tail_f1.png](file:///c:/Users/kushal%20niboriya/Documents/FinDoc/reports/figures/imbalance_ablation_tail_f1.png)

### Key Empirical Findings:
1. **Balanced Class Weights Provide the Largest Tail Improvement**: Inverse-frequency class weights boosted the Tail F1 score from **0.5334 to 0.5968 (+6.34 percentage points)** and overall Macro-F1 from **0.6384 to 0.6685 (+3.01%)**. It forces the gradient updates to treat errors on rare classes with equal gravity.
2. **Focal Loss Excels on Hard Examples Without Sacrificing Head Accuracy**: Focal Loss ($\gamma=2.0$) improved Tail F1 to 0.5651 while simultaneously improving Head F1 to 0.8495 and achieving the highest Micro-F1 (0.7758).
3. **Compound Weighting Leads to Destabilization**: Combining both Focal Loss and strong class weights over-penalizes frequent head classes, degrading Micro-F1 from 0.7710 down to 0.6691.

---

## 5. Critical Engineering Analysis: Classical vs Neural on LEDGAR
An honest, rigorous finding emerges when comparing across all models:
- **Linear SVM (0.8261 Macro-F1)** significantly outperforms FastText (0.7333), TextCNN (0.7543), and DistilBERT (0.7710) when trained without extensive pre-training domain adaptation.
- **Why?**: The LEDGAR dataset contains 100 classes with highly stereotyped lexical formulations (e.g. specific phrases such as *"subject matter hereof"*, *"deemed duly given"*, *"material adverse effect"*, *"waiver of jury trial"*). A high-dimensional TF-IDF unigram+bigram representation (25,000 dimensions) with an L2 margin-maximizer (Linear SVM) directly indexes these exact phrase indicators with zero truncation. In contrast, DistilBERT with `max_len=128` truncates ~31.5% of longer clauses and has to distribute 67M weights across 100 outputs over only 3 epochs.
- This represents a textbook demonstration of the **Occam's Razor principle in NLP**: on structured legal boilerplate classification, a well-tuned linear model on n-gram features can match or exceed standard general-domain transformer encoders.
