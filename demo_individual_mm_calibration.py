"""
Demo: Individual Market Maker Calibration

Fetches data and calibrates for an individual market maker participant:
1. Gemini tick data (as far back as available)
2. Kraken OHLCV (last week, 5-min intervals)
3. Scale arrival rates by 1/1000 (individual MM share)
4. Generate plots and analysis

Author: Yunian Pan
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

print("=" * 80)
print("INDIVIDUAL MARKET MAKER CALIBRATION")
print("=" * 80)

# ============================================================================
# Step 1: Fetch Gemini Tick Data
# ============================================================================

print("\n" + "-" * 80)
print("STEP 1: FETCHING GEMINI TICK DATA")
print("-" * 80)

from calibration.data.gemini_fetcher import GeminiFetcher

gemini = GeminiFetcher()

# Fetch as much data as available (Gemini typically provides last 24-48 hours)
print("\nFetching Gemini BTC-USD trades (max available history)...")
try:
    df_tick = gemini.get_recent_trades(
        symbol='btcusd',
        max_trades=50000,  # Fetch up to 50k trades
    )

    print(f"\n✓ Fetched {len(df_tick):,} trades")
    print(f"  Time span: {pd.to_datetime(df_tick['timestamp'].min(), unit='s')} to {pd.to_datetime(df_tick['timestamp'].max(), unit='s')}")
    curr_time = datetime.now()
    curr_time = curr_time.strftime("%Y-%m-%d %H:%M:%S") 
    # Save for later use
    df_tick.to_parquet(f'gemini_btcusd_full.parquet{curr_time}', index=False)
    print(f"  Saved to: gemini_btcusd_full.parquet")

except Exception as e:
    print(f"\n✗ Error fetching Gemini data: {e}")
    print("  Falling back to existing parquet file...")
    df_tick = pd.read_parquet('notebooks/gemini_btcusd_trades.parquet')

# ============================================================================
# Step 2: Fetch Kraken OHLCV Data
# ============================================================================

print("\n" + "-" * 80)
print("STEP 2: FETCHING KRAKEN OHLCV DATA")
print("-" * 80)

from calibration.data.kraken_fetcher import KrakenFetcher

kraken = KrakenFetcher()

print("\nFetching Kraken BTC/USD 30-min data (last 7 days)...")
try:
    df_ohlcv = kraken.get_ohlcv(
        pair='XXBTZUSD',  # BTC/USD on Kraken
        interval=30,       # 5-minute candles
        days=7,           # Last week
    )

    print(f"\n✓ Fetched {len(df_ohlcv):,} candles")
    print(f"  Time span: {df_ohlcv.index[0]} to {df_ohlcv.index[-1]}")

    # Save for later use
    df_ohlcv.to_csv('kraken_btcusd_5m_7d.csv')
    print(f"  Saved to: kraken_btcusd_5m_7d.csv")

except Exception as e:
    print(f"\n✗ Error fetching Kraken data: {e}")
    print("  Please check your internet connection or API availability")
    sys.exit(1)

# ============================================================================
# Step 3: Calibrate Parameters
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: CALIBRATION")
print("=" * 80)

from calibration.physical.as_calibration import RegimeCalibrator, MicrostructureCalibrator

# Regime calibration from Kraken data
print("\n[1] Regime Calibration (from Kraken 5m data)")
print("-" * 80)

regime_cal = RegimeCalibrator(
    rolling_window=100,  # ~8 hours of 5-min data
    annual_factor=252 * 24 * 12,  # 5-min periods per year
)

regime_params = regime_cal.calibrate(df_ohlcv, price_col='Close')

print(f"\nStable Regime (Low Vol):")
print(f"  σ₀: {regime_params['sigma_stable']:.2%} annualized")
print(f"  μ₀: {regime_params['mu_stable']:+.2%} annualized")

print(f"\nVolatile Regime (High Vol):")
print(f"  σ₁: {regime_params['sigma_volatile']:.2%} annualized")
print(f"  μ₁: {regime_params['mu_volatile']:+.2%} annualized")

print(f"\nRegime Switching:")
print(f"  Base rate: {regime_params['base_transition_rate']:.3f} per day")
print(f"  Avg holding time: {1/regime_params['base_transition_rate']:.1f} days")

# Microstructure calibration from Gemini data
print("\n[2] Microstructure Calibration (from Gemini tick data)")
print("-" * 80)

micro_cal = MicrostructureCalibrator(
    annual_factor=252 * 24 * 60 * 60,  # Seconds in trading year
)

micro_params = micro_cal.calibrate(df_tick)

print(f"\nFull Exchange Arrival Rates:")
print(f"  λ₀ (exchange): {micro_params['lambda_0']:,.0f} per year")
print(f"  κ (sensitivity): {micro_params['kappa']:.3f}")
print(f"  Trades per day: {micro_params['trades_per_day']:,.0f}")

# Scale by 1/1000 for individual market maker
INDIVIDUAL_MM_SHARE = 1/1000

lambda_0_individual = micro_params['lambda_0'] * INDIVIDUAL_MM_SHARE
trades_per_day_individual = micro_params['trades_per_day'] * INDIVIDUAL_MM_SHARE

print(f"\nIndividual Market Maker (1/1000 share):")
print(f"  λ₀ (individual): {lambda_0_individual:,.0f} per year")
print(f"  Trades per day: {trades_per_day_individual:,.2f}")
print(f"  Average trade size: {micro_params['avg_trade_size']:.4f} BTC")

# ============================================================================
# Step 4: Expected Arrivals at Different Spreads
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: EXPECTED ARRIVALS PER TIMESTEP")
print("=" * 80)

dt_seconds = 15
dt_annual = dt_seconds / (252 * 24 * 60 * 60)
kappa = micro_params['kappa']

print(f"\nFor 15-second timestep:")
print(f"\n{'Spread (bps)':>15} | {'λ(δ) per year':>18} | {'E[arrivals/15s]':>18} | {'Decay':>8}")
print("-" * 70)

spread_scenarios = []

for spread_bps in [1, 5, 10, 20, 50, 100]:
    delta = spread_bps / 10000
    decay = np.exp(-kappa * delta)
    lambda_adj = lambda_0_individual * decay
    E_arrivals = lambda_adj * dt_annual

    spread_scenarios.append({
        'spread_bps': spread_bps,
        'lambda_adj': lambda_adj,
        'E_arrivals': E_arrivals,
        'decay': decay,
    })

    print(f"{spread_bps:>15} | {lambda_adj:>18,.0f} | {E_arrivals:>18.4f} | {decay:>7.1%}")

# ============================================================================
# Step 5: Visualizations
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: GENERATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Individual Market Maker Calibration Results', fontsize=14, fontweight='bold')

# Plot 1: Regime Volatility Over Time
ax1 = axes[0, 0]
returns = np.diff(np.log(df_ohlcv['Close'].values))
vol_series = pd.Series(returns).rolling(window=100, min_periods=50).std() * np.sqrt(252*24*12)
vol_series = vol_series.dropna()

regime_labels = regime_params['regime_labels']
# Align times with vol_series length
times = df_ohlcv.index[len(df_ohlcv) - len(vol_series):]

ax1.plot(times, vol_series.values, label='Rolling Vol', alpha=0.7, linewidth=1)
ax1.axhline(regime_params['sigma_stable'], color='green', linestyle='--', label=f"Stable σ={regime_params['sigma_stable']:.1%}")
ax1.axhline(regime_params['sigma_volatile'], color='red', linestyle='--', label=f"Volatile σ={regime_params['sigma_volatile']:.1%}")

# Skip regime background coloring (array alignment issues)

ax1.set_xlabel('Time')
ax1.set_ylabel('Annualized Volatility')
ax1.set_title('Regime Switching Detection (Kraken 5m data)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Trade Arrival Distribution
ax2 = axes[0, 1]
inter_arrivals = np.diff(df_tick['timestamp'].values)
inter_arrivals_clean = inter_arrivals[(inter_arrivals > 0) & (inter_arrivals < 10)]  # Remove outliers

ax2.hist(inter_arrivals_clean, bins=50, alpha=0.7, edgecolor='black')
ax2.axvline(np.mean(inter_arrivals_clean), color='red', linestyle='--', label=f'Mean = {np.mean(inter_arrivals_clean):.3f}s')
ax2.set_xlabel('Inter-arrival Time (seconds)')
ax2.set_ylabel('Frequency')
ax2.set_title(f'Trade Inter-arrival Distribution (Gemini)\n{len(df_tick):,} trades')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Spread Sensitivity Curve
ax3 = axes[1, 0]
spread_range = np.linspace(0, 100, 100)  # 0-100 bps
lambda_curve = lambda_0_individual * np.exp(-kappa * spread_range / 10000)

ax3.plot(spread_range, lambda_curve / 1000, linewidth=2)
ax3.axhline(lambda_0_individual / 1000, color='red', linestyle='--', alpha=0.5, label='Base λ₀')
ax3.set_xlabel('Spread (bps)')
ax3.set_ylabel('Arrival Rate (thousands per year)')
ax3.set_title(f'Spread Sensitivity: λ(δ) = λ₀ · exp(-κδ)\nκ = {kappa:.2f}, Individual MM')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Expected Fills Per Timestep
ax4 = axes[1, 1]
spreads = [s['spread_bps'] for s in spread_scenarios]
arrivals = [s['E_arrivals'] for s in spread_scenarios]

ax4.bar(range(len(spreads)), arrivals, tick_label=spreads, alpha=0.7, edgecolor='black')
ax4.set_xlabel('Spread (bps)')
ax4.set_ylabel('Expected Fills per 15s Timestep')
ax4.set_title(f'Fill Rate vs Spread (Individual MM)\n1/1000 market share')
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (spread, arrival) in enumerate(zip(spreads, arrivals)):
    ax4.text(i, arrival, f'{arrival:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('individual_mm_calibration.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved visualization: individual_mm_calibration.png")

# ============================================================================
# Step 6: Summary and Recommendations
# ============================================================================

print("\n" + "=" * 80)
print("CALIBRATION SUMMARY")
print("=" * 80)

print(f"""
DATA SOURCES:
  - Regime: Kraken BTC/USD 5-min data ({len(df_ohlcv):,} candles, 7 days)
  - Microstructure: Gemini BTC-USD ticks ({len(df_tick):,} trades)

REGIME PARAMETERS (from Kraken):
  - σ_stable:   {regime_params['sigma_stable']:.2%}
  - σ_volatile: {regime_params['sigma_volatile']:.2%}
  - μ_stable:   {regime_params['mu_stable']:+.2%}
  - μ_volatile: {regime_params['mu_volatile']:+.2%}
  - Transition: {regime_params['base_transition_rate']:.3f}/day

MICROSTRUCTURE (Individual MM, 1/1000 share):
  - λ₀: {lambda_0_individual:,.0f} per year ({trades_per_day_individual:.1f}/day)
  - κ: {kappa:.3f}
  - Avg trade: {micro_params['avg_trade_size']:.4f} BTC

EXPECTED FILLS (15s timestep @ 10bps spread):
  - ~{spread_scenarios[2]['E_arrivals']:.3f} fills per step
  - ~{spread_scenarios[2]['E_arrivals'] * 2880:.1f} fills per 12-hour session

RECOMMENDED ENVIRONMENT CONFIG:
  env = make_market_making_env(
      sigma_stable={regime_params['sigma_stable']:.6f},
      sigma_volatile={regime_params['sigma_volatile']:.6f},
      mu_stable={regime_params['mu_stable']:.6f},
      mu_volatile={regime_params['mu_volatile']:.6f},
      base_transition_rate={regime_params['base_transition_rate']:.6f},
      lambda_0={lambda_0_individual:.0f},
      kappa={kappa:.3f},
      xi={0.1:.3f},  # Estimated from vol difference
  )
""")

print("=" * 80)
print("✓ CALIBRATION COMPLETE!")
print("=" * 80)
print("\nFiles generated:")
print("  - gemini_btcusd_full.parquet (tick data)")
print("  - kraken_btcusd_5m_7d.csv (OHLCV data)")
print("  - individual_mm_calibration.png (visualizations)")
print("\nUse these parameters for realistic individual market maker simulation.")
print("=" * 80 + "\n")
