"""
Demo: Regime Calibration and Visualization from Kraken 30-min Data

Calibrates regime-switching parameters and plots price evolution
with regime coloring (green = stable, red = volatile).

Author: Yunian Pan
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from calibration.physical.as_calibration import RegimeCalibrator

print("=" * 80)
print("REGIME CALIBRATION FROM KRAKEN 30-MIN DATA")
print("=" * 80)

# ============================================================================
# Step 1: Load Data
# ============================================================================

print("\n[1] Loading Kraken BTC-USD 30-min OHLCV data...")

df = pd.read_csv('Kraken_btcusd_30m_7d.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

print(f"\n✓ Loaded {len(df):,} candles")
print(f"  Time range: {df.index[0]} to {df.index[-1]}")
print(f"  Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")

# ============================================================================
# Step 2: Calibrate Regime Parameters
# ============================================================================

print("\n[2] Calibrating regime-switching parameters...")

# Use rolling window of 20 periods (10 hours for 30-min candles)
# Annual factor: 30-min intervals → 252 days * 24 hours * 2 intervals/hour
annual_factor = 252 * 24 * 2

calibrator = RegimeCalibrator(
    rolling_window=20,
    annual_factor=annual_factor,
)

regime_params = calibrator.calibrate(df, price_col='Close')

print("\n✓ Calibration complete!")
print(f"\nStable Regime (Low Volatility):")
print(f"  σ₀: {regime_params['sigma_stable']:.4f} ({regime_params['sigma_stable']:.2%} annualized)")
print(f"  μ₀: {regime_params['mu_stable']:+.4f} ({regime_params['mu_stable']:+.2%} annualized)")

print(f"\nVolatile Regime (High Volatility):")
print(f"  σ₁: {regime_params['sigma_volatile']:.4f} ({regime_params['sigma_volatile']:.2%} annualized)")
print(f"  μ₁: {regime_params['mu_volatile']:+.4f} ({regime_params['mu_volatile']:+.2%} annualized)")

print(f"\nRegime Switching:")
print(f"  Base transition rate: {regime_params['base_transition_rate']:.4f} per 30-min period")
print(f"  Avg holding time: {1/regime_params['base_transition_rate']:.1f} periods ({1/regime_params['base_transition_rate']/2:.1f} hours)")

# ============================================================================
# Step 3: Compute Rolling Volatility for Visualization
# ============================================================================

print("\n[3] Computing rolling volatility...")

returns = np.diff(np.log(df['Close'].values))
vol_series = pd.Series(returns).rolling(
    window=20,
    min_periods=10
).std() * np.sqrt(annual_factor)

vol_series = vol_series.dropna()

# Align regime labels with volatility series
regime_labels = regime_params['regime_labels']
valid_len = min(len(vol_series), len(regime_labels))

# Align times
times = df.index[len(df) - valid_len:]
vol_values = vol_series.values[-valid_len:]
regime_labels_aligned = regime_labels[-valid_len:]
prices = df['Close'].iloc[-valid_len:]

# ============================================================================
# Step 4: Create Visualization
# ============================================================================

print("\n[4] Creating visualization...")

fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))
fig.suptitle(
    'BTC-USD (Source: Kraken)',
    fontsize=16,
    fontweight='bold',
    y=0.98
)

# Left y-axis: Price with regime-colored segments
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Price ($)', fontsize=12, color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax1.grid(True, alpha=0.3, linestyle='--', zorder=0)

# Plot price line with different colors for each regime
for i in range(len(times) - 1):
    if regime_labels_aligned[i] == 0:  # Stable regime
        color = 'green'
        label = 'Price (Stable Regime)' if i == 0 or regime_labels_aligned[i-1] != 0 else None
    else:  # Volatile regime
        color = 'red'
        label = 'Price (Volatile Regime)' if i == 0 or regime_labels_aligned[i-1] != 1 else None

    ax1.plot(times[i:i+2], prices.values[i:i+2], linewidth=2.5, color=color,
             label=label, zorder=3, alpha=0.8)

# Right y-axis: Volatility (dashed line)
ax2 = ax1.twinx()
ax2.plot(times, vol_values, linewidth=2, color='blue', linestyle='--',
         label='Rolling Volatility', alpha=0.7, zorder=2)

# Add horizontal lines for regime volatilities
ax2.axhline(
    regime_params['sigma_stable'],
    color='green',
    linestyle=':',
    linewidth=1.5,
    label=f"Stable σ₀ = {regime_params['sigma_stable']:.2%}",
    alpha=0.6,
    zorder=1
)
ax2.axhline(
    regime_params['sigma_volatile'],
    color='red',
    linestyle=':',
    linewidth=1.5,
    label=f"Volatile σ₁ = {regime_params['sigma_volatile']:.2%}",
    alpha=0.6,
    zorder=1
)

ax2.set_ylabel('Annualized Volatility', fontsize=12, color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

# Combine legends
from matplotlib.patches import Patch
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

# Remove duplicate labels
seen_labels = set()
unique_lines = []
unique_labels = []
for line, label in zip(lines1 + lines2, labels1 + labels2):
    if label and label not in seen_labels:
        unique_lines.append(line)
        unique_labels.append(label)
        seen_labels.add(label)

ax1.legend(unique_lines, unique_labels, loc='upper left', fontsize=10, framealpha=0.9)

# Format x-axis
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.xaxis.set_major_locator(mdates.DayLocator())
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()

# Save figure
output_path = 'regime_calibration_kraken_30m.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved visualization: {output_path}")

# ============================================================================
# Step 5: Generate Summary Report
# ============================================================================

print("\n" + "=" * 80)
print("CALIBRATION SUMMARY REPORT")
print("=" * 80)

# Calculate regime statistics
n_stable = np.sum(regime_labels_aligned == 0)
n_volatile = np.sum(regime_labels_aligned == 1)
pct_stable = n_stable / len(regime_labels_aligned) * 100
pct_volatile = n_volatile / len(regime_labels_aligned) * 100

print(f"""
DATA SOURCE:
  - Exchange: Kraken
  - Pair: BTC-USD
  - Interval: 30-minute candles
  - Observations: {len(df):,} candles
  - Time range: {df.index[0].strftime('%Y-%m-%d %H:%M')} to {df.index[-1].strftime('%Y-%m-%d %H:%M')}
  - Duration: {(df.index[-1] - df.index[0]).days} days, {(df.index[-1] - df.index[0]).seconds // 3600} hours

PRICE STATISTICS:
  - Starting price: ${df['Close'].iloc[0]:,.2f}
  - Ending price: ${df['Close'].iloc[-1]:,.2f}
  - Min price: ${df['Close'].min():,.2f}
  - Max price: ${df['Close'].max():,.2f}
  - Price change: {(df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100:+.2f}%

REGIME PARAMETERS:
  ┌─────────────────────┬────────────────┬────────────────┐
  │                     │ Stable (σ₀)    │ Volatile (σ₁)  │
  ├─────────────────────┼────────────────┼────────────────┤
  │ Volatility          │ {regime_params['sigma_stable']:>13.2%} │ {regime_params['sigma_volatile']:>13.2%} │
  │ Drift (μ)           │ {regime_params['mu_stable']:>+13.2%} │ {regime_params['mu_volatile']:>+13.2%} │
  │ Time in regime      │ {pct_stable:>12.1f}% │ {pct_volatile:>12.1f}% │
  └─────────────────────┴────────────────┴────────────────┘

REGIME SWITCHING:
  - Base transition rate: {regime_params['base_transition_rate']:.4f} per 30-min period
  - Daily transition rate: {regime_params['base_transition_rate'] * 48:.4f} per day
  - Average holding time: {1/regime_params['base_transition_rate']:.1f} periods ({1/regime_params['base_transition_rate']/2:.1f} hours)
  - Expected switches per day: {regime_params['base_transition_rate'] * 48:.2f}

RECOMMENDED MODEL CONFIGURATION:

  env = make_market_making_env(
      # Regime parameters (from Kraken calibration)
      sigma_stable={regime_params['sigma_stable']:.6f},      # {regime_params['sigma_stable']:.2%} annualized
      sigma_volatile={regime_params['sigma_volatile']:.6f},    # {regime_params['sigma_volatile']:.2%} annualized
      mu_stable={regime_params['mu_stable']:.6f},         # {regime_params['mu_stable']:+.2%} annualized
      mu_volatile={regime_params['mu_volatile']:.6f},      # {regime_params['mu_volatile']:+.2%} annualized
      base_transition_rate={regime_params['base_transition_rate'] * 48:.6f},  # per day (converted from 30-min rate)

      # Microstructure parameters (from Gemini tick data)
      lambda_0=...,      # Base arrival intensity
      kappa=...,         # Spread sensitivity
      xi=...,            # Predator cost coefficient
  )

INTERPRETATION:
  - The stable regime has {regime_params['sigma_stable']:.1%} annualized volatility
  - The volatile regime has {regime_params['sigma_volatile']:.1%} annualized volatility
  - Volatility ratio: {regime_params['sigma_volatile'] / regime_params['sigma_stable']:.2f}x higher in volatile regime
  - The market spent {pct_stable:.1f}% of time in stable regime, {pct_volatile:.1f}% in volatile regime
  - On average, regime switches occur every {1/regime_params['base_transition_rate']/2:.1f} hours
""")

print("=" * 80)
print("✓ CALIBRATION COMPLETE!")
print("=" * 80)
print(f"\nGenerated file: {output_path}")
print("\nUse these parameters for your market making environment configuration.")
print("=" * 80 + "\n")
