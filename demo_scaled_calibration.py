"""
Demo: Scaled Calibration for Realistic Simulation

The raw Gemini tick data represents ultra-HFT activity (496M trades/year).
This script shows how to scale parameters for different trading scenarios.

Author: Yunian Pan
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from calibration.physical.as_calibration import quick_calibrate_from_yfinance

print("=" * 80)
print("SCALED CALIBRATION FOR REALISTIC MARKET MAKING")
print("=" * 80)

# Get base calibration from yfinance
result = quick_calibrate_from_yfinance(
    ticker='BTC-USD',
    start_date='2024-01-01',
    end_date='2025-01-01',
    interval='1d',
    gamma=0.01,
)

print("\nBASE CALIBRATION (from yfinance):")
print(f"  σ_stable: {result.sigma_stable:.2%}")
print(f"  σ_volatile: {result.sigma_volatile:.2%}")
print(f"  Transition rate: {result.base_transition_rate:.3f}/day")

print("\n" + "=" * 80)
print("ARRIVAL RATE SCENARIOS")
print("=" * 80)

scenarios = {
    "Ultra-HFT (Gemini actual)": {
        "lambda_0": 496_815_709,
        "kappa": 10.0,
        "description": "Market maker on centralized exchange (22 trades/sec)",
    },
    "Institutional MM": {
        "lambda_0": 1_000_000,
        "kappa": 5.0,
        "description": "Professional market maker (~4K trades/day)",
    },
    "Active Retail": {
        "lambda_0": 100_000,
        "kappa": 3.0,
        "description": "Active retail trader (~400 trades/day)",
    },
    "Conservative Retail": {
        "lambda_0": 25_200,
        "kappa": 1.5,
        "description": "Conservative strategy (~100 trades/day)",
    },
}

dt_annual = 15 / (252 * 24 * 60 * 60)  # 15s timestep

for scenario_name, params in scenarios.items():
    lambda_0 = params['lambda_0']
    kappa = params['kappa']

    print(f"\n{scenario_name}")
    print(f"  {params['description']}")
    print(f"  λ₀ = {lambda_0:,} per year")
    print(f"  κ = {kappa}")

    # Calculate arrivals at different spreads
    print(f"\n  Arrivals per 15s timestep:")
    print(f"    {'Spread':>10} | {'Expected fills':>15}")
    print(f"    {'-'*10}-+-{'-'*15}")

    for spread_bps in [5, 10, 20, 50]:
        delta = spread_bps / 10000
        lambda_adj = lambda_0 * np.exp(-kappa * delta)
        E_arrivals = lambda_adj * dt_annual

        print(f"    {spread_bps:>10} | {E_arrivals:>15.2f}")

print("\n" + "=" * 80)
print("RECOMMENDED CONFIGURATIONS")
print("=" * 80)

print("""
1. FOR TESTING / DEBUGGING:
   env = make_market_making_env(
       lambda_0=25_200,      # 100 trades/day
       kappa=1.5,
       xi=0.02,
   )
   # Expect: ~0.03 fills per 15s step @ 10bps spread

2. FOR REALISTIC INSTITUTIONAL MM:
   env = make_market_making_env(
       lambda_0=1_000_000,   # 4K trades/day
       kappa=5.0,
       xi=0.05,
   )
   # Expect: ~1.15 fills per 15s step @ 10bps spread

3. FOR HFT RESEARCH (matches Gemini data):
   env = make_market_making_env(
       lambda_0=496_815_709,  # From calibration
       kappa=10.0,
       xi=0.1,
   )
   # Expect: ~339 fills per 15s step @ 10bps spread
   # WARNING: Very compute-intensive!

4. CALIBRATED REGIME + SCALED ARRIVALS:
   env = make_market_making_env(
       # Use calibrated regime params
       sigma_stable={result.sigma_stable:.6f},
       sigma_volatile={result.sigma_volatile:.6f},
       mu_stable={result.mu_stable:.6f},
       mu_volatile={result.mu_volatile:.6f},
       base_transition_rate={result.base_transition_rate:.6f},

       # But scale down arrivals
       lambda_0=100_000,  # Conservative
       kappa=3.0,
       xi=0.05,
   )
""")

print("\n" + "=" * 80)
print("WHY SCALE DOWN?")
print("=" * 80)

print("""
The Gemini tick data represents ultra-high-frequency market making where:
- Market makers compete with microsecond latency
- Spreads are extremely tight (0.01% or less)
- Order flow is dominated by HFT algos

For academic research or backtesting strategies:
- Use lower λ₀ to match your actual trading frequency
- Adjust κ based on your spread sensitivity assumptions
- Keep calibrated regime params (they capture BTC volatility regimes well)

The calibration framework is CORRECT - it just reflects the reality
that Gemini's order book is dominated by professional market makers!
""")

print("=" * 80)
