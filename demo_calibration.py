"""
Demo: BTC Market Making Calibration

Demonstrates the two-part calibration framework:
1. Regime calibration from OHLCV data (yfinance)
2. Microstructure calibration from Gemini tick data

Then runs a simulation using calibrated parameters vs defaults.

Author: Yunian Pan
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from calibration.physical.as_calibration import (
    calibrate_from_csv_and_parquet,
    quick_calibrate_from_yfinance,
)
from calibration.data.yfinance_fetcher import YFinanceFetcher
from market_making import make_market_making_env, get_optimal_profile, PnLSimulator

print("=" * 80)
print("BTC MARKET MAKING CALIBRATION DEMO")
print("=" * 80)

# ============================================================================
# Part 1: Fetch and Save OHLCV Data
# ============================================================================

print("\n" + "-" * 80)
print("FETCHING BTC OHLCV DATA")
print("-" * 80)

fetcher = YFinanceFetcher()

# Fetch 1 year of BTC data at daily intervals
# Note: yfinance intraday data has time limits, use daily for full year
df_ohlcv = fetcher.get_history(
    ticker='BTC-USD',
    start='2024-01-01',
    end='2025-01-01',
    interval='1d',
)

print(f"\nFetched {len(df_ohlcv)} OHLCV observations")
print(f"Period: {df_ohlcv['Date'].iloc[0]} to {df_ohlcv['Date'].iloc[-1]}")
print(f"\nFirst 5 rows:")
print(df_ohlcv.head())

# Save to CSV for calibrate_from_csv_and_parquet()
# Set Date as index for calibration code
ohlcv_path = 'btc_ohlcv_daily.csv'
df_ohlcv.set_index('Date').to_csv(ohlcv_path)
print(f"\nSaved to {ohlcv_path}")

# ============================================================================
# Part 2: Quick Calibration from yfinance only (no tick data)
# ============================================================================

print("\n" + "-" * 80)
print("QUICK CALIBRATION (OHLCV only)")
print("-" * 80)

result_quick = quick_calibrate_from_yfinance(
    ticker='BTC-USD',
    start_date='2024-01-01',
    end_date='2025-01-01',
    interval='1d',
    gamma=0.01,
)

print(result_quick)

# ============================================================================
# Part 3: Full Calibration with Tick Data
# ============================================================================

print("\n" + "-" * 80)
print("FULL CALIBRATION (OHLCV + Tick Data)")
print("-" * 80)

tick_path = 'notebooks/gemini_btcusd_trades.parquet'

try:
    result_full = calibrate_from_csv_and_parquet(
        ohlcv_path=ohlcv_path,
        tick_path=tick_path,
        gamma=0.01,
    )

    print(result_full)

    # Export for environment
    env_config = result_full.to_env_config()
    print("\n" + "-" * 80)
    print("ENVIRONMENT CONFIG")
    print("-" * 80)
    for key, val in env_config.items():
        print(f"{key:25s}: {val}")

except FileNotFoundError as e:
    print(f"\nWARNING: Tick data not found ({e})")
    print("Using quick calibration result for demo...")
    result_full = result_quick
    env_config = result_full.to_env_config()

# ============================================================================
# Part 4: Comparison - Calibrated vs Default Parameters
# ============================================================================

print("\n" + "=" * 80)
print("SIMULATION COMPARISON: Calibrated vs Default")
print("=" * 80)

# Default environment config
env_config_default = {
    'S_0': 50000.0,
    'gamma': 0.01,
    'kappa': 1.5,
    'xi': 0.02,
    'max_steps': 2880,
    'dt_minutes': 0.25,
    'macro_freq': 120,
}

# Calibrated environment config
env_config_calibrated = {
    'S_0': df_ohlcv['Close'].iloc[-1],  # Use latest price
    'gamma': 0.01,
    'kappa': env_config.get('kappa', 1.5),
    'xi': env_config.get('xi', 0.02),
    'max_steps': 2880,
    'dt_minutes': 0.25,
    'macro_freq': 120,
    # BTC process params
    'sigma_stable': env_config['sigma_stable'],
    'sigma_volatile': env_config['sigma_volatile'],
    'mu_stable': env_config['mu_stable'],
    'mu_volatile': env_config['mu_volatile'],
    'base_transition_rate': env_config['base_transition_rate'],
    'lambda_0': env_config.get('lambda_0', 100 * 252),
}

print("\n" + "-" * 80)
print("DEFAULT PARAMETERS")
print("-" * 80)
for key, val in env_config_default.items():
    print(f"{key:25s}: {val}")

print("\n" + "-" * 80)
print("CALIBRATED PARAMETERS")
print("-" * 80)
for key, val in env_config_calibrated.items():
    print(f"{key:25s}: {val}")

# ============================================================================
# Part 5: Run Short Simulation
# ============================================================================

print("\n" + "-" * 80)
print("RUNNING SHORT SIMULATION (100 paths)")
print("-" * 80)

# Shorter simulation for demo
env_config_default_short = env_config_default.copy()
env_config_default_short['max_steps'] = 500

env_config_calibrated_short = env_config_calibrated.copy()
env_config_calibrated_short['max_steps'] = 500

# Create strategy profiles
profile_default = get_optimal_profile(
    gamma=env_config_default['gamma'],
    kappa=env_config_default['kappa'],
    xi=env_config_default['xi'],
)
profile_default.name = "Default Parameters"

profile_calibrated = get_optimal_profile(
    gamma=env_config_calibrated['gamma'],
    kappa=env_config_calibrated['kappa'],
    xi=env_config_calibrated['xi'],
)
profile_calibrated.name = "Calibrated Parameters"

# Run default
print("\n1. Default parameters...")
simulator_default = PnLSimulator(
    env_config=env_config_default_short,
    n_paths=100,
    verbose=False,
)
results_default = simulator_default.run([profile_default])
stats_default = simulator_default.compute_statistics(results_default)

# Run calibrated
print("2. Calibrated parameters...")
simulator_calibrated = PnLSimulator(
    env_config=env_config_calibrated_short,
    n_paths=100,
    verbose=False,
)
results_calibrated = simulator_calibrated.run([profile_calibrated])
stats_calibrated = simulator_calibrated.compute_statistics(results_calibrated)

# ============================================================================
# Part 6: Comparison Results
# ============================================================================

print("\n" + "=" * 80)
print("CALIBRATION IMPACT ON PERFORMANCE")
print("=" * 80)

default_stats = stats_default["Default Parameters"]
calib_stats = stats_calibrated["Calibrated Parameters"]

print(f"\n{'Metric':<20s} | {'Default':>15s} | {'Calibrated':>15s} | {'Delta':>12s}")
print("-" * 70)

metrics = [
    ('Mean PnL', 'mean_pnl', lambda x: f"${x:,.2f}"),
    ('Std PnL', 'std_pnl', lambda x: f"${x:,.2f}"),
    ('Sharpe Ratio', 'sharpe', lambda x: f"{x:.3f}"),
    ('Win Rate', 'win_rate', lambda x: f"{x*100:.1f}%"),
    ('P5', 'p5_pnl', lambda x: f"${x:,.2f}"),
    ('P95', 'p95_pnl', lambda x: f"${x:,.2f}"),
]

for metric_name, key, fmt in metrics:
    default_val = default_stats[key]
    calib_val = calib_stats[key]

    if key == 'win_rate':
        delta = f"{(calib_val - default_val)*100:+.1f}pp"
    elif key == 'sharpe':
        delta = f"{calib_val - default_val:+.3f}"
    else:
        delta = f"${calib_val - default_val:+,.2f}"

    print(f"{metric_name:<20s} | {fmt(default_val):>15s} | {fmt(calib_val):>15s} | {delta:>12s}")

# ============================================================================
# Key Insights
# ============================================================================

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

print(f"\n1. REGIME VOLATILITY:")
print(f"   Calibrated σ_stable:   {result_full.sigma_stable:.2%}")
print(f"   Calibrated σ_volatile: {result_full.sigma_volatile:.2%}")
print(f"   Ratio: {result_full.sigma_volatile / result_full.sigma_stable:.2f}x")

print(f"\n2. DRIFT PARAMETERS:")
print(f"   Calibrated μ_stable:   {result_full.mu_stable:+.2%}")
print(f"   Calibrated μ_volatile: {result_full.mu_volatile:+.2%}")

print(f"\n3. PREDATOR COST:")
print(f"   Calibrated ξ: {result_full.xi_estimate:.4f}")
print(f"   Max drift bound: {result_full.max_drift_bound:+.6f}")

if hasattr(result_full, 'lambda_0') and result_full.lambda_0 != 100 * 252:
    print(f"\n4. MICROSTRUCTURE:")
    print(f"   Base arrival rate λ₀: {result_full.lambda_0:,.0f} per year")
    print(f"   Sensitivity κ: {result_full.kappa:.3f}")
    print(f"   Avg trade size: {result_full.avg_trade_size:.4f} BTC")
    print(f"   Trades per day: {result_full.trades_per_day:,.0f}")

print(f"\n5. PERFORMANCE IMPACT:")
pnl_delta = calib_stats['mean_pnl'] - default_stats['mean_pnl']
sharpe_delta = calib_stats['sharpe'] - default_stats['sharpe']
print(f"   ΔPnL: ${pnl_delta:+,.2f}")
print(f"   ΔSharpe: {sharpe_delta:+.3f}")

print("\n" + "=" * 80)
print("✓ CALIBRATION DEMO COMPLETE!")
print("=" * 80)
print(f"\nCalibrated parameters are ready for use in market making simulations.")
print(f"Use result.to_env_config() to export parameters for make_market_making_env().")
print("=" * 80 + "\n")
