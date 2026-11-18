"""
Unit tests for the FinDoc Pipeline and Entity Extraction integration.
"""

from src.pipeline import FinDocPipeline
from src.extract import extract_entities


def test_entity_extraction():
    text = "On January 15, 2024, Acme Corp LLC shall pay $10,500,000 within thirty (30) days."
    ents = extract_entities(text)
    assert len(ents["amounts"]) >= 1
    assert "$10,500,000" in ents["amounts"][0]
    assert len(ents["dates"]) >= 1
    assert "January 15, 2024" in ents["dates"][0]
    assert len(ents["durations"]) >= 1
    assert any("thirty" in d for d in ents["durations"])


def test_pipeline_clause_analysis():
    # Fast test without heavy retrieval indexing
    pipeline = FinDocPipeline(confidence_threshold=0.50, enable_retrieval=False)
    clause = "This Agreement shall be governed by, and construed in accordance with, the laws of the State of New York."
    res = pipeline.analyze_clause(clause)
    
    assert "top_label" in res
    assert "top_3_predictions" in res
    assert len(res["top_3_predictions"]) == 3
    assert "confidence" in res
    assert res["top_label"] in ["Governing Laws", "Applicable Laws"]
    assert res["status"] in ["Confident", "Uncertain (Human Review Recommended)"]


if __name__ == "__main__":
    test_entity_extraction()
    test_pipeline_clause_analysis()
    print("All pipeline tests passed successfully!")
