"""
Fine-tuning DistilBERT for Clause Classification on LEDGAR.
Configuration:
- Model: distilbert-base-uncased
- Max sequence length: 128
- Mixed precision: fp16 (AMP)
- Batch size: 16 (per device)
- Epochs: 3
- Optimizer: AdamW, lr=3e-5, weight_decay=0.01, linear warmup

Logs:
- Train and validation loss per epoch
- Validation and Test Macro-F1, Micro-F1, Accuracy
- Peak VRAM allocated
- Training time and inference latency
Saves best model weights for downstream evaluation and CLI.
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score

from src.data import load_ledgar_offline, get_dataset_metadata
from src.features import get_default_preprocessor
from src.preprocess import deduplicate_corpus

CHECKPOINT_DIR = Path("models/checkpoints/distilbert_ledgar")
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 3e-5


class TransformerClauseDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = MAX_LEN):
        self.labels = labels
        print(f"Tokenizing {len(texts):,} clauses (max_len={max_len})...")
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def train_distilbert(save_checkpoint: bool = True):
    print("=" * 70)
    print("FINE-TUNING DISTILBERT ON LEDGAR (FP16 MIXED PRECISION)")
    print("=" * 70)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DistilBERT fine-tuning.")

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print("\n[1/5] Loading LEDGAR dataset...")
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
    pipe = get_default_preprocessor()
    print("Preprocessing raw clauses with frozen pipeline...")
    train_texts = pipe.transform_batch(ds["train"]["text"])
    val_texts = pipe.transform_batch(ds["validation"]["text"])
    test_texts = pipe.transform_batch(ds["test"]["text"])

    train_texts, train_labels, _ = deduplicate_corpus(train_texts, ds["train"]["label"])
    val_labels = ds["validation"]["label"]
    test_labels = ds["test"]["label"]

    print(f"Preprocessed: Train={len(train_texts):,}, Val={len(val_texts):,}, Test={len(test_texts):,}")

    # 2. Tokenize & Create Loaders
    print("\n[2/5] Initializing tokenizer & datasets...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = TransformerClauseDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_ds = TransformerClauseDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    test_ds = TransformerClauseDataset(test_texts, test_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=True)

    # 3. Model & Optimizer
    print(f"\n[3/5] Loading {MODEL_NAME} for {num_classes} classes...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    torch.cuda.reset_peak_memory_stats(device)
    t_train_start = time.perf_counter()

    epoch_logs = []
    best_val_macro = -1.0
    best_model_state = None

    print(f"\n[4/5] Training for {EPOCHS} epochs ({total_steps:,} total steps)...")
    for epoch in range(1, EPOCHS + 1):
        t_epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()

            if step % 500 == 0 or step == len(train_loader):
                print(f"  Epoch {epoch}/{EPOCHS} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_epoch_loss = running_loss / len(train_loader)
        epoch_sec = time.perf_counter() - t_epoch_start

        # Validation Pass
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item()

                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        labels_all = list(range(num_classes))
        val_macro = f1_score(val_targets, val_preds, labels=labels_all, average="macro", zero_division=0)
        val_acc = accuracy_score(val_targets, val_preds)

        print(f"\n--> Epoch {epoch} Summary ({epoch_sec/60:.2f} min):")
        print(f"    Train Loss: {train_epoch_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Macro-F1: {val_macro:.4f} | Val Acc: {val_acc:.4f}\n")

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": round(train_epoch_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_macro_f1": round(float(val_macro), 4),
            "val_acc": round(float(val_acc), 4),
            "epoch_time_sec": round(epoch_sec, 2),
        })

        if val_macro > best_val_macro:
            best_val_macro = val_macro
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if save_checkpoint:
                print(f"    Saving new best model checkpoint to {CHECKPOINT_DIR}...")
                model.save_pretrained(CHECKPOINT_DIR)
                tokenizer.save_pretrained(CHECKPOINT_DIR)

    total_train_time = time.perf_counter() - t_train_start
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    peak_vram_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

    # 4. Evaluation on Test Set using best model
    print("\n[5/5] Evaluating Best DistilBERT Checkpoint on Test Set...")
    model.load_state_dict(best_model_state)
    model.eval()

    test_preds, test_targets = [], []
    t_test_start = time.perf_counter()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(labels.cpu().numpy())

    test_eval_time = time.perf_counter() - t_test_start
    latency_ms_per_sample = (test_eval_time / len(test_targets)) * 1000

    test_macro = f1_score(test_targets, test_preds, labels=labels_all, average="macro", zero_division=0)
    test_micro = f1_score(test_targets, test_preds, labels=labels_all, average="micro", zero_division=0)
    test_acc = accuracy_score(test_targets, test_preds)
    per_class_f1 = f1_score(test_targets, test_preds, labels=labels_all, average=None, zero_division=0)

    head_f1 = float(np.mean([per_class_f1[i] for i in class_tiers["head"]]))
    mid_f1 = float(np.mean([per_class_f1[i] for i in class_tiers["mid"]]))
    tail_f1 = float(np.mean([per_class_f1[i] for i in class_tiers["tail"]]))

    print(f"\n============================================================")
    print(f"DISTILBERT TEST SET RESULTS")
    print(f"============================================================")
    print(f"Test Macro-F1:       {test_macro:.4f} (Headline)")
    print(f"Test Micro-F1:       {test_micro:.4f}")
    print(f"Test Accuracy:       {test_acc:.4f}")
    print(f"Tier Performance:    Head={head_f1:.4f} | Mid={mid_f1:.4f} | Tail={tail_f1:.4f}")
    print(f"Inference Latency:   {latency_ms_per_sample:.3f} ms / sample ({1000/latency_ms_per_sample:.1f} samples/sec)")
    print(f"Peak VRAM:           {peak_vram_gb:.2f} GB allocated ({peak_vram_reserved_gb:.2f} GB reserved)")
    print(f"Total Train Time:    {total_train_time/60:.2f} minutes")
    print(f"============================================================")

    results = {
        "model_name": "DistilBERT (Fine-Tuned, 3 Epochs, fp16)",
        "parameters": total_params,
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "train_time_sec": round(total_train_time, 2),
        "train_time_min": round(total_train_time / 60, 2),
        "peak_vram_allocated_gb": round(peak_vram_gb, 2),
        "peak_vram_reserved_gb": round(peak_vram_reserved_gb, 2),
        "latency_ms_per_sample": round(latency_ms_per_sample, 3),
        "epoch_logs": epoch_logs,
        "best_val_macro_f1": round(float(best_val_macro), 4),
        "test_metrics": {
            "macro_f1": round(float(test_macro), 4),
            "micro_f1": round(float(test_micro), 4),
            "accuracy": round(float(test_acc), 4),
            "tier_f1": {
                "head_f1": round(head_f1, 4),
                "mid_f1": round(mid_f1, 4),
                "tail_f1": round(tail_f1, 4),
            }
        },
        "per_class_f1": {label_names[i]: round(float(per_class_f1[i]), 4) for i in range(num_classes)}
    }

    return results


if __name__ == "__main__":
    train_distilbert()
