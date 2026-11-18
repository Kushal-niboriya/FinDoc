"""
Script to extract and display 8-10 before/after preprocessing examples
using actual LEDGAR training samples.
"""

from src.data import load_ledgar_offline
from src.preprocess import PreprocessingPipeline


def run_examples():
    ds = load_ledgar_offline()
    train_ds = ds["train"]
    label_names = train_ds.features["label"].names

    # Select 8 diverse clauses with numbering, dates, amounts, parties
    indices = [0, 15, 42, 105, 230, 450, 789, 1024, 1540]
    
    # Configure specialized pipeline
    pipeline_specialized = PreprocessingPipeline(
        lowercase=True,
        remove_stopwords=False,
        lemmatize=False,
        remove_numbering=True,
        mask_money=True,
        mask_dates=True,
        mask_durations=True,
        normalize_parties=True,
        collapse_boilerplate=True
    )

    examples = []
    print("=" * 80)
    print("8-10 BEFORE / AFTER LEGAL PREPROCESSING EXAMPLES (ACTUAL LEDGAR DATA)")
    print("=" * 80)

    for i, idx in enumerate(indices, start=1):
        raw_text = train_ds[idx]["text"]
        label_idx = train_ds[idx]["label"]
        label_name = label_names[label_idx]

        processed_text = pipeline_specialized.transform_text(raw_text)

        print(f"\n--- EXAMPLE {i} [Class: {label_name}] ---")
        print(f"BEFORE:\n{raw_text[:220]}...")
        print(f"AFTER:\n{processed_text[:220]}...")
        
        examples.append({
            "index": idx,
            "class": label_name,
            "raw": raw_text,
            "processed": processed_text
        })

    return examples


if __name__ == "__main__":
    run_examples()
