"""
FinDoc Command Line Interface (CLI).
Usage:
    python -m src.cli analyze path/to/contract.txt [--threshold 0.50] [--output results.json]
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

from src.pipeline import FinDocPipeline


def print_contract_analysis(result: Dict[str, Any]):
    """Renders human-readable formatted table for the command line."""
    print("\n" + "=" * 95)
    print(f"FINDOC CONTRACT CLAUSE INTELLIGENCE REPORT")
    print("=" * 95)
    print(f"Total Clauses Detected:      {result['total_clauses']}")
    print(f"Confident Classifications:   {result['confident_clauses']} ({result['confident_clauses']/result['total_clauses']*100:.1f}%)")
    print(f"Uncertain Classifications:   {result['uncertain_clauses']} ({result['uncertain_clauses']/result['total_clauses']*100:.1f}%) [Threshold: {result['confidence_threshold_used']}]")
    print("-" * 95)

    # Contract-wide entities
    ents = result["contract_entities"]
    print("KEY CONTRACT ENTITIES EXTRACTED:")
    print(f"  Amounts:   {', '.join(ents['amounts']) if ents['amounts'] else 'None detected'}")
    print(f"  Dates:     {', '.join(ents['dates']) if ents['dates'] else 'None detected'}")
    print(f"  Durations: {', '.join(ents['durations']) if ents['durations'] else 'None detected'}")
    print(f"  Parties:   {', '.join(ents['parties'][:6]) if ents['parties'] else 'None detected'}")
    print("=" * 95)

    print(f"\n{'ID':<4} | {'HEADER':<24} | {'CLASSIFICATION':<26} | {'ENTITIES FOUND':<20} | {'STATUS'}")
    print("-" * 95)

    for item in result["clauses"]:
        c_id = item["clause_id"]
        header = (item["header"][:22] + "..") if len(item["header"]) > 24 else item["header"]
        ana = item["analysis"]
        top_pred = ana["top_3_predictions"][0]
        label_str = f"{top_pred['label']} ({top_pred['confidence_pct']}%)"

        ent_summary = []
        if ana["entities"]["amounts"]:
            ent_summary.append(f"{len(ana['entities']['amounts'])} amt")
        if ana["entities"]["dates"]:
            ent_summary.append(f"{len(ana['entities']['dates'])} date")
        if ana["entities"]["durations"]:
            ent_summary.append(f"{len(ana['entities']['durations'])} dur")
        if ana["entities"]["parties"]:
            ent_summary.append(f"{len(ana['entities']['parties'])} party")
        ent_str = ", ".join(ent_summary) if ent_summary else "none"

        status_tag = "[UNCERTAIN]" if ana["is_uncertain"] else "[CONFIDENT]"
        print(f"{c_id:<4} | {header:<24} | {label_str:<26} | {ent_str:<20} | {status_tag}")

        # Top 3 predictions breakdown
        preds_str = " | ".join([f"{p['label']}: {p['confidence_pct']}%" for p in ana["top_3_predictions"]])
        print(f"     Ranked Predictions: {preds_str}")

        # Similar clause
        if ana["similar_clauses"]:
            top_sim = ana["similar_clauses"][0]
            print(f"     Top Precedent: [{top_sim['category']} (score: {top_sim['score']:.3f})] {top_sim['text_snippet']}")
        print("-" * 95)


def main():
    parser = argparse.ArgumentParser(description="FinDoc: Clause Intelligence for Financial & Legal Contracts")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    analyze_parser = subparsers.add_parser("analyze", help="Segment, classify, retrieve, and extract from contract")
    analyze_parser.add_argument("contract_path", type=str, help="Path to contract plain-text file")
    analyze_parser.add_argument("--threshold", type=float, default=0.50, help="Confidence abstention threshold (default: 0.50)")
    analyze_parser.add_argument("--output", "-o", type=str, default=None, help="Save structured JSON output to path")
    analyze_parser.add_argument("--no-retrieval", action="store_true", help="Disable reference clause retrieval")

    args = parser.parse_args()

    if args.command == "analyze":
        contract_path = Path(args.contract_path)
        if not contract_path.exists():
            print(f"Error: Contract file '{contract_path}' does not exist.")
            sys.exit(1)

        with open(contract_path, "r", encoding="utf-8") as f:
            contract_text = f.read()

        pipeline = FinDocPipeline(
            confidence_threshold=args.threshold,
            enable_retrieval=not args.no_retrieval
        )
        results = pipeline.analyze_contract(contract_text)

        print_contract_analysis(results)

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved structured analysis to {out_p}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
