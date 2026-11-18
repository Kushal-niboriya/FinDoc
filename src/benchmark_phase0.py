"""
Phase 0 feasibility benchmark:
Runs a 200-step training loop on DistilBERT (max_len=128, fp16=True, batch_size=16)
to profile peak GPU VRAM and seconds-per-step on NVIDIA RTX 3050 Laptop GPU (4 GB).
Estimates full training run times and flags runs exceeding 1 hour.
"""

import os
import time
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

from src.data import load_ledgar_offline


def run_benchmark(
    model_name: str = "distilbert-base-uncased",
    max_steps: int = 200,
    batch_size: int = 16,
    max_len: int = 128,
    lr: float = 5e-5,
    save_result_path: str = "results/phase0_benchmark.json"
):
    print("=" * 60)
    print("PHASE 0 BENCHMARK: DISTILBERT FP16 VRAM & LATENCY PROFILING")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU is strictly required for this benchmark.")

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    free_vram_gb, _ = torch.cuda.mem_get_info()
    free_vram_gb /= (1024 ** 3)
    
    print(f"Device: {gpu_name}")
    print(f"Total VRAM: {total_vram_gb:.2f} GB | Currently Free: {free_vram_gb:.2f} GB")

    # Load offline dataset
    print("\n[1/5] Loading LEDGAR dataset from disk...")
    ds = load_ledgar_offline()
    train_ds = ds["train"]
    num_train = len(train_ds)
    num_classes = len(train_ds.features["label"].names)
    print(f"Loaded {num_train} train examples, {num_classes} classes.")

    # Load tokenizer and model
    print(f"\n[2/5] Initializing tokenizer & model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes
    )
    model.to(device)
    model.train()

    # Prepare small slice for the 200-step benchmark (200 * 16 = 3200 samples)
    benchmark_samples = train_ds.select(range(max_steps * batch_size))
    
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors=None
        )

    print("\n[3/5] Tokenizing benchmark slice...")
    tokenized_ds = benchmark_samples.map(
        tokenize_fn,
        batched=True,
        batch_size=256,
        remove_columns=["text"]
    )
    tokenized_ds.set_format("torch")
    loader = DataLoader(tokenized_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # Warmup step
    print("\n[4/5] Executing 5 warm-up steps...")
    warmup_loader = iter(loader)
    for _ in range(5):
        batch = next(warmup_loader)
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Reset peak memory tracker
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    print(f"\n[5/5] Running timed benchmark: {max_steps} steps (batch_size={batch_size}, max_len={max_len}, fp16=True)...")
    start_time = time.perf_counter()

    step = 0
    running_loss = 0.0
    for batch in loader:
        if step >= max_steps:
            break
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        step += 1
        if step % 50 == 0:
            print(f"  Step {step}/{max_steps} - current loss: {loss.item():.4f}")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    sec_per_step = elapsed / max_steps
    steps_per_sec = max_steps / elapsed

    peak_vram_bytes = torch.cuda.max_memory_allocated()
    peak_vram_gb = peak_vram_bytes / (1024 ** 3)
    peak_vram_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)

    # Calculate full-run estimates
    steps_per_epoch = num_train // batch_size  # 60,000 / 16 = 3,750 steps
    time_1_epoch_sec = steps_per_epoch * sec_per_step
    time_1_epoch_min = time_1_epoch_sec / 60
    time_3_epochs_min = time_1_epoch_min * 3

    # Feasibility check against 1 hour limit
    exceeds_1_hour = time_3_epochs_min > 60

    results = {
        "hardware": {
            "gpu_name": gpu_name,
            "total_vram_gb": round(total_vram_gb, 2),
            "initial_free_vram_gb": round(free_vram_gb, 2),
        },
        "config": {
            "model_name": model_name,
            "batch_size": batch_size,
            "max_len": max_len,
            "fp16": True,
            "steps_benchmarked": max_steps,
        },
        "benchmark_metrics": {
            "elapsed_seconds": round(elapsed, 2),
            "sec_per_step": round(sec_per_step, 4),
            "steps_per_sec": round(steps_per_sec, 2),
            "peak_vram_allocated_gb": round(peak_vram_gb, 2),
            "peak_vram_reserved_gb": round(peak_vram_reserved_gb, 2),
            "safe_headroom_gb": round(total_vram_gb - peak_vram_reserved_gb, 2),
        },
        "projections": {
            "dataset_train_size": num_train,
            "steps_per_epoch": steps_per_epoch,
            "est_1_epoch_minutes": round(time_1_epoch_min, 2),
            "est_3_epochs_minutes": round(time_3_epochs_min, 2),
            "est_3_epochs_hours": round(time_3_epochs_min / 60, 2),
            "exceeds_1_hour_flag": exceeds_1_hour,
        },
        "planned_experiments_estimates": {
            "DistilBERT_full_3epochs_bs16": {
                "steps": steps_per_epoch * 3,
                "est_minutes": round(time_3_epochs_min, 2),
                "flag_exceeds_1hr": exceeds_1_hour,
            },
            "DistilBERT_subset_20k_3epochs_bs16": {
                "steps": (20000 // batch_size) * 3,
                "est_minutes": round((20000 // batch_size) * 3 * sec_per_step / 60, 2),
                "flag_exceeds_1hr": False,
            },
            "MiniLM_L6_full_3epochs_bs16": {
                # MiniLM L6 is ~half the parameters of DistilBERT (22M vs 66M), ~1.4x faster
                "steps": steps_per_epoch * 3,
                "est_minutes": round(time_3_epochs_min * 0.7, 2),
                "flag_exceeds_1hr": (time_3_epochs_min * 0.7) > 60,
            }
        }
    }

    Path(save_result_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Peak VRAM Allocated: {peak_vram_gb:.2f} GB / {total_vram_gb:.2f} GB")
    print(f"Peak VRAM Reserved:  {peak_vram_reserved_gb:.2f} GB")
    print(f"Latency:             {sec_per_step:.4f} sec/step ({steps_per_sec:.2f} steps/s)")
    print(f"Est. 1 Epoch Time:   {time_1_epoch_min:.2f} minutes")
    print(f"Est. 3 Epochs Time:  {time_3_epochs_min:.2f} minutes ({time_3_epochs_min/60:.2f} hours)")
    print(f"Exceeds 1 hour limit: {exceeds_1_hour}")
    print("=" * 60)
    print(f"Saved benchmark metrics to {save_result_path}")
    return results


if __name__ == "__main__":
    run_benchmark()
