"""
PyTorch Neural Models for FinDoc:
1. FastText-style EmbeddingBag Classifier
2. Multi-Filter TextCNN (Kim, 2014)

Trained on GPU with fp16 mixed precision and CPU fallback.
Records: parameter count, peak VRAM, training time, inference latency, Macro-F1.
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from src.data import load_ledgar_offline, get_dataset_metadata
from src.features import get_default_preprocessor
from src.preprocess import deduplicate_corpus, tokenize_simple

MAX_VOCAB_SIZE = 25000
MAX_SEQ_LEN = 128
EMBED_DIM = 128
NUM_CLASSES = 100
PAD_IDX = 0
UNK_IDX = 1


class Vocabulary:
    def __init__(self, max_size=MAX_VOCAB_SIZE):
        self.max_size = max_size
        self.w2i = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}
        self.i2w = {PAD_IDX: "<pad>", UNK_IDX: "<unk>"}

    def fit(self, tokenized_texts: List[List[str]]):
        from collections import Counter
        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)
        
        most_common = counter.most_common(self.max_size - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.w2i[word] = idx
            self.i2w[idx] = word

    def encode(self, tokens: List[str], max_len: int = MAX_SEQ_LEN) -> Tuple[List[int], int]:
        ids = [self.w2i.get(t, UNK_IDX) for t in tokens[:max_len]]
        length = len(ids)
        if length == 0:
            ids = [UNK_IDX]
            length = 1
        return ids, length


class ClauseDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Vocabulary, max_len: int = MAX_SEQ_LEN):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.padded_seqs = []
        self.lengths = []

        for text in texts:
            tokens = tokenize_simple(text)
            ids, length = vocab.encode(tokens, max_len)
            # Pad to max_len
            padded = ids + [PAD_IDX] * (max_len - len(ids))
            self.padded_seqs.append(padded)
            self.lengths.append(length)

        self.padded_seqs = torch.tensor(self.padded_seqs, dtype=torch.long)
        self.lengths = torch.tensor(self.lengths, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.padded_seqs[idx],
            "length": self.lengths[idx],
            "label": self.labels[idx]
        }


class FastTextEmbeddingBag(nn.Module):
    """FastText-style averaging bag-of-words classifier."""
    def __init__(self, vocab_size: int, embed_dim: int = EMBED_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean", padding_idx=PAD_IDX)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, lengths=None):
        # input_ids: (B, L)
        emb = self.embedding(input_ids)
        logits = self.fc(emb)
        return logits


class TextCNN(nn.Module):
    """Multi-filter 1D CNN for text classification (Kim, 2014)."""
    def __init__(self, vocab_size: int, embed_dim: int = EMBED_DIM, num_classes: int = NUM_CLASSES, num_filters: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in (3, 4, 5)
        ])
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(len(self.convs) * num_filters, num_classes)

    def forward(self, input_ids, lengths=None):
        # input_ids: (B, L)
        x = self.embedding(input_ids) # (B, L, D)
        x = x.transpose(1, 2) # (B, D, L) for Conv1d
        
        pooled_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x)) # (B, num_filters, L - k + 1)
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.shape[2]).squeeze(2) # (B, num_filters)
            pooled_outputs.append(pooled)
            
        feat = torch.cat(pooled_outputs, dim=1) # (B, 3 * num_filters)
        feat = self.dropout(feat)
        logits = self.fc(feat)
        return logits


def train_eval_neural_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-3,
    class_tiers: Dict = None,
    label_names: List[str] = None
) -> Dict[str, Any]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--- Training {model_name} on {device} ({param_count:,} params, {epochs} epochs) ---")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_val_macro = -1.0
    best_weights = None
    t_train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        step_count = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            lengths = batch["length"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
                logits = model(input_ids, lengths)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            step_count += 1

        avg_loss = total_loss / step_count

        # Validation evaluation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                lengths = batch["length"].to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
                    logits = model(input_ids, lengths)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch["label"].numpy())

        val_macro = f1_score(val_targets, val_preds, labels=list(range(len(label_names))), average="macro", zero_division=0)
        val_acc = accuracy_score(val_targets, val_preds)
        print(f"  Epoch {epoch}/{epochs} | Train Loss: {avg_loss:.4f} | Val Macro-F1: {val_macro:.4f} | Val Acc: {val_acc:.4f}")

        if val_macro > best_val_macro:
            best_val_macro = val_macro
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time_sec = time.perf_counter() - t_train_start

    # Peak VRAM
    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    # Load best model weights for test evaluation
    model.load_state_dict(best_weights)
    model.eval()

    # Inference latency benchmark on test set
    test_preds, test_targets = [], []
    t_inf_start = time.perf_counter()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["length"].to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
                logits = model(input_ids, lengths)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(batch["label"].numpy())

    inf_total_sec = time.perf_counter() - t_inf_start
    latency_ms_per_sample = (inf_total_sec / len(test_targets)) * 1000

    test_macro = f1_score(test_targets, test_preds, labels=list(range(len(label_names))), average="macro", zero_division=0)
    test_micro = f1_score(test_targets, test_preds, labels=list(range(len(label_names))), average="micro", zero_division=0)
    test_acc = accuracy_score(test_targets, test_preds)
    per_class_f1 = f1_score(test_targets, test_preds, labels=list(range(len(label_names))), average=None, zero_division=0)

    # Tier breakdowns
    head_f1 = float(np.mean([per_class_f1[i] for i in class_tiers["head"]]))
    mid_f1 = float(np.mean([per_class_f1[i] for i in class_tiers["mid"]]))
    tail_f1 = float(np.mean([per_class_f1[i] for i in class_tiers["tail"]]))

    print(f"  --> {model_name} Test Results: Macro-F1: {test_macro:.4f} | Micro-F1: {test_micro:.4f} | Head: {head_f1:.4f} | Mid: {mid_f1:.4f} | Tail: {tail_f1:.4f}")
    print(f"      Latency: {latency_ms_per_sample:.3f} ms/sample | Peak VRAM: {peak_vram_mb:.1f} MB | Train Time: {train_time_sec:.1f}s")

    return {
        "model_name": model_name,
        "parameters": param_count,
        "train_time_sec": round(train_time_sec, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "latency_ms_per_sample": round(latency_ms_per_sample, 3),
        "val_macro_f1": round(float(best_val_macro), 4),
        "test_metrics": {
            "macro_f1": round(float(test_macro), 4),
            "micro_f1": round(float(test_micro), 4),
            "accuracy": round(float(test_acc), 4),
            "tier_f1": {
                "head_f1": round(head_f1, 4),
                "mid_f1": round(mid_f1, 4),
                "tail_f1": round(tail_f1, 4),
            }
        }
    }


def run_neural_experiments():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")

    ds = load_ledgar_offline()
    meta = get_dataset_metadata(ds)
    label_names = meta["label_names"]
    num_classes = len(label_names)

    train_counts = np.bincount(ds["train"]["label"], minlength=num_classes)
    sorted_classes = np.argsort(-train_counts)
    class_tiers = {
        "head": sorted_classes[:15].tolist(),
        "mid": sorted_classes[15:50].tolist(),
        "tail": sorted_classes[50:].tolist(),
    }

    # Preprocess
    print("Preprocessing datasets with frozen pipeline...")
    pipe = get_default_preprocessor()
    train_texts = pipe.transform_batch(ds["train"]["text"])
    val_texts = pipe.transform_batch(ds["validation"]["text"])
    test_texts = pipe.transform_batch(ds["test"]["text"])

    # Deduplicate train
    train_texts, train_labels, _ = deduplicate_corpus(train_texts, ds["train"]["label"])

    # Fit Vocabulary
    print("Building vocabulary from training tokens...")
    tokenized_train = [tokenize_simple(t) for t in train_texts]
    vocab = Vocabulary(max_size=MAX_VOCAB_SIZE)
    vocab.fit(tokenized_train)
    print(f"Vocabulary size: {len(vocab.w2i):,} tokens")

    # Datasets & Loaders
    batch_size = 64
    train_ds = ClauseDataset(train_texts, train_labels, vocab)
    val_ds = ClauseDataset(val_texts, ds["validation"]["label"], vocab)
    test_ds = ClauseDataset(test_texts, ds["test"]["label"], vocab)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    results = {}

    # Model 1: FastText EmbeddingBag
    fasttext_model = FastTextEmbeddingBag(len(vocab.w2i))
    res_ft = train_eval_neural_model(
        fasttext_model, "FastText EmbeddingBag",
        train_loader, val_loader, test_loader,
        device=device, epochs=5, lr=2e-3,
        class_tiers=class_tiers, label_names=label_names
    )
    results["FastText EmbeddingBag"] = res_ft

    # Model 2: TextCNN
    textcnn_model = TextCNN(len(vocab.w2i))
    res_cnn = train_eval_neural_model(
        textcnn_model, "TextCNN (1D Multi-Filter)",
        train_loader, val_loader, test_loader,
        device=device, epochs=5, lr=1e-3,
        class_tiers=class_tiers, label_names=label_names
    )
    results["TextCNN"] = res_cnn

    return results


if __name__ == "__main__":
    run_neural_experiments()
