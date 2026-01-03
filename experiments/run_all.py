#!/usr/bin/env python
"""
Run All Market Making Experiments
==================================

This script runs the complete experimental pipeline:
1. Regime calibration (Kraken 30-min data)
2. Microstructure calibration (Gemini tick data)
3. Counterfactual simulation (Vanilla vs Equilibrium AS)

All results are saved to the results/ directory.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_script(script_path, description):
    """Run a Python script and handle errors."""
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print("="*80)
    print(f"Script: {script_path}\n")

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Script not found: {script_path}")
        return False

def main():
    """Run all experiments in sequence."""

    print("="*80)
    print("MARKET MAKING EXPERIMENTS - FULL PIPELINE")
    print("="*80)
    print("\nThis will run:")
    print("  1. BTC Regime Calibration (Kraken 30-min data)")
    print("  2. Gemini Microstructure Calibration (tick data)")
    print("  3. Adversarial AS Counterfactual Simulation (1000 paths)")
    print("\nEstimated time: 5-10 minutes")
    print("="*80)

    # Get the experiments directory
    experiments_dir = Path(__file__).parent

    # Create results directory if it doesn't exist
    results_dir = experiments_dir / "results"
    results_dir.mkdir(exist_ok=True)
    print(f"\n✓ Results directory: {results_dir}")

    # Define experiments to run
    experiments = [
        (
            experiments_dir / "calibration" / "btc_regime_calibration.py",
            "BTC Regime Calibration"
        ),
        (
            experiments_dir / "calibration" / "gemini_microstructure_calibration.py",
            "Gemini Microstructure Calibration"
        ),
        (
            experiments_dir / "adversarial_as" / "demo_counterfactual_simulation.py",
            "Adversarial AS Counterfactual Simulation"
        ),
    ]

    # Run each experiment
    results = []
    for script_path, description in experiments:
        success = run_script(str(script_path), description)
        results.append((description, success))

    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)

    print("\nResults:")
    all_success = True
    for description, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {description}")
        if not success:
            all_success = False

    print("\nOutput files in results/:")
    if results_dir.exists():
        for file in sorted(results_dir.glob("*")):
            print(f"  - {file.name}")

    if all_success:
        print("\n🎉 All experiments completed successfully!")
        return 0
    else:
        print("\n⚠️  Some experiments failed. Check output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
