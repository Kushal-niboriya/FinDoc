"""
Unit tests for the Contract Clause Segmentation engine.
"""

from src.pipeline import segment_contract


def test_section_header_segmentation():
    contract = """
Section 1. Definitions.
Here are the definitions of the terms used herein.

Section 2. Representations and Warranties.
The Company represents and warrants that it is duly incorporated.

Section 3. Governing Law.
This agreement shall be governed by Delaware law.
"""
    clauses = segment_contract(contract)
    assert len(clauses) == 3, f"Expected 3 segmented clauses, got {len(clauses)}"
    assert "Section 1" in clauses[0]["header"]
    assert "Section 2" in clauses[1]["header"]
    assert "Section 3" in clauses[2]["header"]


def test_paragraph_fallback_segmentation():
    contract = """
First major paragraph containing contractual commitments between parties regarding indemnification.

Second major paragraph specifying that notices must be sent via registered airmail.
"""
    clauses = segment_contract(contract)
    assert len(clauses) == 2, f"Expected 2 segmented clauses, got {len(clauses)}"


def test_empty_contract():
    clauses = segment_contract("")
    assert len(clauses) == 0


if __name__ == "__main__":
    test_section_header_segmentation()
    test_paragraph_fallback_segmentation()
    test_empty_contract()
    print("All segmentation tests passed successfully!")
