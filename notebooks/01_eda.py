"""
Interactive EDA script for FinDoc (notebooks/01_eda.py).
Run with: python notebooks/01_eda.py
"""

from src.eda import analyze_dataset

if __name__ == "__main__":
    print("Executing FinDoc Exploratory Data Analysis...")
    stats = analyze_dataset()
    print("EDA Complete. Check results/eda_stats.json and reports/figures/")
