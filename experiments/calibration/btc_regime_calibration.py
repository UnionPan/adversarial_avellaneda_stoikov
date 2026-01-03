"""
BTC Regime Calibration from Kraken 30-min Data
==============================================

This script calibrates regime-switching parameters from Kraken BTC-USD 30-minute OHLCV data.

Uses K-means clustering on rolling volatility to identify two regimes:
- Stable regime (low volatility)
- Volatile regime (high volatility)

Estimates transition rates between regimes for use in the double-layer game framework.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datetime import datetime
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', '..'))

print("="*80)
print("BTC REGIME CALIBRATION - KRAKEN 30-MIN DATA")
print("="*80)

# ============================================================================
# Step 1: Load Data
# ============================================================================

print("\n[1] Loading Kraken 30-min OHLCV data...")

# Get absolute path to project root (two levels up from this script)
script_dir = Path(__file__).parent.resolve()  # experiments/calibration/
project_root = script_dir.parent.parent  # project root

# Try to load from project data directory
data_paths = [
    project_root / 'kraken_btcusd_30m_7d.csv',
    script_dir.parent / 'kraken_btcusd_30m_7d.csv',  # experiments/
    project_root / 'data' / 'kraken_btcusd_30m_7d.csv',
]

df = None
for path in data_paths:
    try:
        df = pd.read_csv(str(path))
        # Rename Date column to timestamp for consistency
        if 'Date' in df.columns:
            df = df.rename(columns={'Date': 'timestamp'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Standardize column names to lowercase
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        print(f"  ✓ Loaded from: {path}")
        break
    except FileNotFoundError:
        continue

if df is None:
    print("  ✗ Data file not found. Please ensure kraken_btcusd_30m_7d.csv exists.")
    print("    Expected paths:")
    for path in data_paths:
        print(f"      - {path}")
    sys.exit(1)

print(f"  Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"  Observations: {len(df)}")

# ============================================================================
# Step 2: Calculate Rolling Volatility
# ============================================================================

print("\n[2] Calculating rolling volatility...")

# Calculate log returns
df['log_return'] = np.log(df['close'] / df['close'].shift(1))

# Rolling window for volatility (4 hours = 8 periods of 30 min)
window = 8
df['rolling_vol'] = df['log_return'].rolling(window=window).std() * np.sqrt(365.25 * 24 * 2)  # Annualized

# Drop NaN values
df = df.dropna().reset_index(drop=True)

print(f"  Window: {window} periods ({window*0.5:.1f} hours)")
print(f"  Valid observations: {len(df)}")

# ============================================================================
# Step 3: Regime Identification via K-means
# ============================================================================

print("\n[3] Identifying regimes via K-means clustering...")

# Prepare data for clustering
X = df['rolling_vol'].values.reshape(-1, 1)

# K-means with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df['regime'] = kmeans.fit_predict(X)

# Ensure regime 0 is stable (low vol), regime 1 is volatile (high vol)
cluster_means = df.groupby('regime')['rolling_vol'].mean()
if cluster_means[0] > cluster_means[1]:
    # Swap labels
    df['regime'] = 1 - df['regime']
    cluster_means = df.groupby('regime')['rolling_vol'].mean()

# Compute regime statistics
vol_stable = df[df['regime'] == 0]['rolling_vol'].mean()
vol_volatile = df[df['regime'] == 1]['rolling_vol'].mean()

print(f"  Regime 0 (Stable): σ = {vol_stable:.4f} ({vol_stable*100:.2f}% annualized)")
print(f"  Regime 1 (Volatile): σ = {vol_volatile:.4f} ({vol_volatile*100:.2f}% annualized)")
print(f"  Volatility ratio: {vol_volatile/vol_stable:.2f}x")

# ============================================================================
# Step 4: Estimate Transition Rates
# ============================================================================

print("\n[4] Estimating regime transition rates...")

# Count transitions
transitions = np.diff(df['regime'])
n_01 = np.sum(transitions == 1)  # Stable → Volatile
n_10 = np.sum(transitions == -1)  # Volatile → Stable

# Count time spent in each regime
n_stable = np.sum(df['regime'] == 0)
n_volatile = np.sum(df['regime'] == 1)

# Estimate transition rates (per 30-min period)
lambda_01 = n_01 / n_stable if n_stable > 0 else 0
lambda_10 = n_10 / n_volatile if n_volatile > 0 else 0

# Convert to per-day rates (48 periods per day)
periods_per_day = 48
lambda_01_day = lambda_01 * periods_per_day
lambda_10_day = lambda_10 * periods_per_day

print(f"  Transitions 0→1: {n_01}")
print(f"  Transitions 1→0: {n_10}")
print(f"  Time in regime 0: {n_stable} periods ({n_stable/len(df)*100:.1f}%)")
print(f"  Time in regime 1: {n_volatile} periods ({n_volatile/len(df)*100:.1f}%)")
print(f"\n  Transition rate λ₀₁: {lambda_01_day:.2f} per day (avg holding: {1/lambda_01_day*24:.1f} hours)")
print(f"  Transition rate λ₁₀: {lambda_10_day:.2f} per day (avg holding: {1/lambda_10_day*24:.1f} hours)")

# Base transition rate (average)
base_rate = (lambda_01_day + lambda_10_day) / 2
print(f"\n  Base transition rate: {base_rate:.2f} per day (avg holding: {1/base_rate*24:.1f} hours)")

# ============================================================================
# Step 5: Visualization
# ============================================================================

print("\n[5] Creating visualization...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Top panel: Price evolution colored by regime
ax1 = axes[0]
for i in range(len(df) - 1):
    color = 'red' if df.iloc[i]['regime'] == 1 else 'green'
    ax1.plot([df.iloc[i]['timestamp'], df.iloc[i+1]['timestamp']],
             [df.iloc[i]['close'], df.iloc[i+1]['close']],
             color=color, linewidth=1.5, alpha=0.7)

ax1.set_ylabel('BTC Price ($)', fontsize=12)
ax1.set_title('BTC Price Evolution (Green = Stable, Red = Volatile)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(['Stable Regime', 'Volatile Regime'], loc='upper left')

# Bottom panel: Rolling volatility with regime boundaries
ax2 = axes[1]
stable_data = df[df['regime'] == 0]
volatile_data = df[df['regime'] == 1]

ax2.scatter(stable_data['timestamp'], stable_data['rolling_vol'],
           c='green', s=10, alpha=0.5, label=f'Stable (σ={vol_stable:.3f})')
ax2.scatter(volatile_data['timestamp'], volatile_data['rolling_vol'],
           c='red', s=10, alpha=0.5, label=f'Volatile (σ={vol_volatile:.3f})')

# Add mean lines
ax2.axhline(vol_stable, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax2.axhline(vol_volatile, color='red', linestyle='--', linewidth=2, alpha=0.7)

ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Annualized Volatility', fontsize=12)
ax2.set_title('Rolling Volatility by Regime', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save to results directory (create if doesn't exist)
results_dir = script_dir.parent / 'results'
results_dir.mkdir(exist_ok=True)
output_path = results_dir / 'regime_calibration_kraken_30m.png'
plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {output_path.relative_to(project_root)}")

# ============================================================================
# Step 6: Export Calibrated Parameters
# ============================================================================

print("\n[6] Exporting calibrated parameters...")

params = {
    'SIGMA_STABLE': vol_stable,
    'SIGMA_VOLATILE': vol_volatile,
    'TRANSITION_RATE_01': lambda_01_day,
    'TRANSITION_RATE_10': lambda_10_day,
    'BASE_TRANSITION_RATE': base_rate,
    'DATA_START': str(df['timestamp'].min()),
    'DATA_END': str(df['timestamp'].max()),
    'N_OBSERVATIONS': len(df),
    'PCT_TIME_STABLE': n_stable / len(df) * 100,
    'PCT_TIME_VOLATILE': n_volatile / len(df) * 100,
}

params_df = pd.DataFrame([params])
csv_path = results_dir / 'regime_parameters.csv'
params_df.to_csv(str(csv_path), index=False)
print(f"  ✓ Saved: {csv_path.relative_to(project_root)}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("REGIME CALIBRATION COMPLETE")
print("="*80)

print("\nCALIBRATED PARAMETERS:")
print(f"  σ_stable  = {vol_stable:.6f}  ({vol_stable*100:.2f}% annualized)")
print(f"  σ_volatile = {vol_volatile:.6f}  ({vol_volatile*100:.2f}% annualized)")
print(f"  Base transition rate = {base_rate:.2f} per day")
print(f"  Avg regime holding time = {1/base_rate*24:.1f} hours")

print("\nUSAGE IN SIMULATION:")
print("  SIGMA_STABLE = {:.6f}".format(vol_stable))
print("  SIGMA_VOLATILE = {:.6f}".format(vol_volatile))
print("  BASE_TRANSITION_RATE = {:.1f}  # per day".format(base_rate))

print("\n" + "="*80)
