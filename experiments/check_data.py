#!/usr/bin/env python
"""
Data Files Checker
==================

Verifies that all required data files are present and accessible.
"""

import os
from pathlib import Path
import pandas as pd

def check_file(path, description):
    """Check if a file exists and can be read."""
    print(f"\nChecking: {description}")
    print(f"  Path: {path}")

    if not os.path.exists(path):
        print(f"  Status: ✗ NOT FOUND")
        return False

    # Try to read the file
    try:
        if path.endswith('.csv'):
            df = pd.read_csv(path)
            print(f"  Status: ✓ FOUND")
            print(f"  Rows: {len(df):,}")
            if 'timestamp' in df.columns:
                print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        elif path.endswith('.parquet'):
            df = pd.read_parquet(path)
            print(f"  Status: ✓ FOUND")
            print(f"  Rows: {len(df):,}")
            if 'timestamp' in df.columns:
                print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        else:
            print(f"  Status: ✓ FOUND (not validated)")

        return True
    except Exception as e:
        print(f"  Status: ✗ ERROR reading file")
        print(f"  Error: {e}")
        return False

def main():
    print("="*80)
    print("DATA FILES CHECKER")
    print("="*80)

    # Get project root (parent of experiments directory)
    experiments_dir = Path(__file__).parent
    project_root = experiments_dir.parent

    print(f"\nProject root: {project_root}")

    # Define required data files
    data_files = [
        (project_root / "kraken_btcusd_30m_7d.csv", "Kraken 30-min OHLCV data"),
        (project_root / "gemini_btcusd_full.parquet", "Gemini tick data (root)"),
        (project_root / "notebooks" / "gemini_btcusd_full.parquet", "Gemini tick data (notebooks)"),
    ]

    results = []
    for path, description in data_files:
        found = check_file(str(path), description)
        results.append((description, found))

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    all_found = True
    for description, found in results:
        status = "✓" if found else "✗"
        print(f"  {status} {description}")
        if not found:
            all_found = False

    if all_found:
        print("\n🎉 All data files found! Ready to run experiments.")
    else:
        print("\n⚠️  Some data files are missing.")
        print("\nRequired data files:")
        print("  1. kraken_btcusd_30m_7d.csv - Kraken 30-minute OHLCV data")
        print("  2. gemini_btcusd_full.parquet - Gemini tick data")
        print("\nThese should be in the project root directory or notebooks/ folder.")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
