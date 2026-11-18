"""
Data ingestion and loading module for FinDoc.
Dataset: LEDGAR (LexGLUE benchmark)
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from datasets import DatasetDict, load_dataset, load_from_disk


DEFAULT_RAW_DIR = Path("data/raw/ledgar")


def download_ledgar(save_dir: Path | str = DEFAULT_RAW_DIR) -> DatasetDict:
    """
    Robustly download the LEDGAR dataset from Hugging Face and persist to disk.
    Tries primary load_dataset('coastalcph/lex_glue', 'ledgar'),
    then falls back to alternative repo or direct parquet URLs if necessary.
    """
    save_dir = Path(save_dir)
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    
    dataset: Optional[DatasetDict] = None
    source_used = None

    # Strategy 1: coastalcph/lex_glue
    try:
        print("[data.py] Attempting primary source: load_dataset('coastalcph/lex_glue', 'ledgar')...")
        dataset = load_dataset("coastalcph/lex_glue", "ledgar")
        source_used = "coastalcph/lex_glue (ledgar config)"
        print("[data.py] Primary source succeeded!")
    except Exception as e1:
        print(f"[data.py] Primary source failed: {e1}")

    # Strategy 2: lex_glue alias
    if dataset is None:
        try:
            print("[data.py] Attempting fallback source 1: load_dataset('lex_glue', 'ledgar')...")
            dataset = load_dataset("lex_glue", "ledgar")
            source_used = "lex_glue (ledgar config)"
            print("[data.py] Fallback source 1 succeeded!")
        except Exception as e2:
            print(f"[data.py] Fallback source 1 failed: {e2}")

    # Strategy 3: Direct parquet files from Hugging Face coastalcph/lex_glue repository
    if dataset is None:
        try:
            print("[data.py] Attempting fallback source 2: Direct parquet stream from huggingface.co...")
            base_url = "https://huggingface.co/datasets/coastalcph/lex_glue/resolve/main/data"
            data_files = {
                "train": f"{base_url}/ledgar-train.parquet",
                "validation": f"{base_url}/ledgar-validation.parquet",
                "test": f"{base_url}/ledgar-test.parquet",
            }
            dataset = load_dataset("parquet", data_files=data_files)
            source_used = "direct parquet files (coastalcph/lex_glue)"
            print("[data.py] Direct parquet fallback succeeded!")
        except Exception as e3:
            print(f"[data.py] Fallback source 2 failed: {e3}")
            raise RuntimeError(f"All download strategies failed for LEDGAR dataset: {e3}")

    print(f"[data.py] Successfully fetched dataset using: {source_used}")
    print(f"[data.py] Saving dataset to disk at {save_dir}...")
    dataset.save_to_disk(str(save_dir))
    print("[data.py] Saved to disk successfully.")
    
    # Store source info
    with open(save_dir / "source_info.txt", "w", encoding="utf-8") as f:
        f.write(f"Source: {source_used}\n")

    return dataset


def load_ledgar_offline(data_dir: Path | str = DEFAULT_RAW_DIR) -> DatasetDict:
    """
    Load LEDGAR dataset strictly offline from disk with HF_DATASETS_OFFLINE=1.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Offline dataset path {data_dir} does not exist. Call download_ledgar() first.")

    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    dataset = load_from_disk(str(data_dir))
    return dataset


def get_dataset_metadata(dataset: DatasetDict) -> Dict[str, Any]:
    """
    Extract summary metadata: split sizes, label names, field names, label count.
    """
    splits = list(dataset.keys())
    split_sizes = {split: len(dataset[split]) for split in splits}
    sample_split = dataset[splits[0]]
    field_names = sample_split.column_names
    
    label_feature = sample_split.features.get("label")
    if hasattr(label_feature, "names"):
        label_names = label_feature.names
    else:
        # If ClassLabel was not preserved, inspect unique labels in train
        all_labels = set(sample_split["label"])
        label_names = [str(i) for i in sorted(all_labels)]

    return {
        "splits": splits,
        "split_sizes": split_sizes,
        "total_examples": sum(split_sizes.values()),
        "field_names": field_names,
        "num_classes": len(label_names),
        "label_names_sample": label_names[:10],
        "label_names": label_names,
    }


if __name__ == "__main__":
    print("[src/data.py] Starting LEDGAR download & verification...")
    download_ledgar()
    
    print("\n[src/data.py] Verifying offline loading...")
    offline_ds = load_ledgar_offline()
    meta = get_dataset_metadata(offline_ds)
    
    print("\n=== LEDGAR DATASET METADATA ===")
    print("Splits:", meta["splits"])
    print("Split sizes:", meta["split_sizes"])
    print("Total examples:", meta["total_examples"])
    print("Field names:", meta["field_names"])
    print("Number of classes:", meta["num_classes"])
    print("First 10 label names:", meta["label_names_sample"])
