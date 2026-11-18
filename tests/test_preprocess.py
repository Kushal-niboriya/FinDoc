"""
Unit tests for the Preprocessing Pipeline.
"""

from src.preprocess import PreprocessingPipeline, deduplicate_corpus


def test_money_masking():
    pipe = PreprocessingPipeline(mask_money=True, lowercase=True)
    text = "Borrower shall repay $15,000,000 in cash."
    out = pipe.transform_text(text)
    assert "[money]" in out, f"Expected [money] in output, got: {out}"


def test_date_masking():
    pipe = PreprocessingPipeline(mask_dates=True, lowercase=True)
    text = "Commencing on January 15, 2024 and terminating on 12/31/2025."
    out = pipe.transform_text(text)
    assert "[date]" in out, f"Expected [date] in output, got: {out}"


def test_duration_masking():
    pipe = PreprocessingPipeline(mask_durations=True, lowercase=True)
    text = "Within thirty (30) business days or 180 calendar days."
    out = pipe.transform_text(text)
    assert "[duration]" in out, f"Expected [duration] in output, got: {out}"


def test_numbering_removal():
    pipe = PreprocessingPipeline(remove_numbering=True, lowercase=True)
    text = "Section 12.04(b) Governing Law. In accordance with Section 1.1."
    out = pipe.transform_text(text)
    assert "[ref]" in out, f"Expected [ref] in output, got: {out}"


def test_deduplicate_corpus():
    texts = [
        "This agreement shall be governed by Delaware law.",
        "THIS AGREEMENT SHALL BE GOVERNED BY DELAWARE LAW. ",
        "Different clause."
    ]
    labels = [1, 1, 2]
    deduped_texts, deduped_labels, removed = deduplicate_corpus(texts, labels)
    assert len(deduped_texts) == 2, f"Expected 2 unique texts, got {len(deduped_texts)}"
    assert removed == 1, f"Expected 1 removed duplicate, got {removed}"


if __name__ == "__main__":
    test_money_masking()
    test_date_masking()
    test_duration_masking()
    test_numbering_removal()
    test_deduplicate_corpus()
    print("All preprocessing tests passed successfully!")
