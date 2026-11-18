"""
Unit test for data ingestion and strict offline loading of LEDGAR.
"""

import os
from src.data import load_ledgar_offline, get_dataset_metadata


def test_offline_loading():
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    ds = load_ledgar_offline("data/raw/ledgar")
    assert ds is not None, "Dataset failed to load offline"
    
    meta = get_dataset_metadata(ds)
    assert set(meta["splits"]) == {"train", "validation", "test"}, f"Unexpected splits: {meta['splits']}"
    assert meta["split_sizes"]["train"] == 60000, f"Expected 60000 train samples, got {meta['split_sizes']['train']}"
    assert meta["split_sizes"]["validation"] == 10000, f"Expected 10000 val samples, got {meta['split_sizes']['validation']}"
    assert meta["split_sizes"]["test"] == 10000, f"Expected 10000 test samples, got {meta['split_sizes']['test']}"
    assert meta["num_classes"] == 100, f"Expected 100 classes, got {meta['num_classes']}"
    assert "text" in meta["field_names"]
    assert "label" in meta["field_names"]
    print("\n[test_offline_loading] Passed all assertions successfully!")


if __name__ == "__main__":
    test_offline_loading()
