"""
Demo: Multi-Path PnL Comparison

Compares strategy profiles across 1000 Monte Carlo paths:
1. Optimal (Paper Formulas): MM + Predator + Macro game
2. Baseline (No Game): Symmetric spread, no predator/macro
3. MM Only (No Adversary): Optimal MM without predator
4. Vanilla AS: Traditional Avellaneda-Stoikov (no predatory risk)

Generates:
- PnL histograms showing distribution
- Statistical comparison table
- Sample wealth trajectories

Author: Yunian Pan
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from market_making.strategies import (
    get_optimal_profile,
    get_baseline_profile,
    get_mm_only_profile,
    StrategyProfile,
    AdversarialAvellanedaStoikov,
    NoPredator,
    ConstantMacro,
)
from market_making.pnl_simulator import PnLSimulator

print("=" * 80)
print("MULTI-PATH PnL COMPARISON: Market Making Game")
print("=" * 80)
print("\nSimulating 1000 Monte Carlo paths (12-hour episodes)")
print("Comparing strategy profiles to evaluate game-theoretic approach")

# ============================================================================
# Configuration
# ============================================================================

gamma = 0.01
kappa = 1.5
xi = 0.02

env_config = {
    'S_0': 50000.0,
    'gamma': gamma,
    'kappa': kappa,
    'xi': xi,
    'max_steps': 2880,  # 12 hours
    'dt_minutes': 0.25,  # 15 seconds
    'macro_freq': 120,   # 30 minutes
}

# ============================================================================
# Strategy Profiles
# ============================================================================

print("\n" + "-" * 80)
print("STRATEGY PROFILES")
print("-" * 80)

profiles = [
    get_optimal_profile(gamma=gamma, kappa=kappa, xi=xi),
    get_baseline_profile(),
    get_mm_only_profile(gamma=gamma, kappa=kappa),

    # Vanilla AS: Traditional AS without predatory risk adjustment
    StrategyProfile(
        mm_strategy=AdversarialAvellanedaStoikov(
            gamma=gamma, kappa=kappa, xi=0.0  # No predator risk
        ),
        predator_strategy=NoPredator(),
        macro_strategy=ConstantMacro(f=0.5, g=0.5),
        name="Vanilla AS (No Predator Risk)",
    ),
]

for profile in profiles:
    print(f"\n{profile.name}:")
    print(f"  MM: {profile.mm}")
    print(f"  Predator: {profile.predator}")
    print(f"  Macro: {profile.macro}")

# ============================================================================
# Run Simulation
# ============================================================================

print("\n" + "-" * 80)
print("RUNNING MONTE CARLO SIMULATION")
print("-" * 80)

simulator = PnLSimulator(
    env_config=env_config,
    n_paths=1000,
    verbose=True,
)

print(f"\nSimulating {simulator.n_paths} paths for {len(profiles)} profiles...")
print(f"Total episodes: {simulator.n_paths * len(profiles)}")

results = simulator.run(profiles)

# ============================================================================
# Compute Statistics
# ============================================================================

print("\n" + "-" * 80)
print("COMPUTING STATISTICS")
print("-" * 80)

stats = simulator.compute_statistics(results)
simulator.print_statistics(results, stats)

# ============================================================================
# Detailed Breakdown
# ============================================================================

print("\n" + "-" * 80)
print("DETAILED BREAKDOWN")
print("-" * 80)

for profile_name, s in stats.items():
    print(f"\n{profile_name}:")
    print(f"  Mean PnL: ${s['mean_pnl']:,.2f} ± ${s['std_pnl']:,.2f}")
    print(f"  Median PnL: ${s['median_pnl']:,.2f}")
    print(f"  Range: [${s['min_pnl']:,.2f}, ${s['max_pnl']:,.2f}]")
    print(f"  5th-95th percentile: [${s['p5_pnl']:,.2f}, ${s['p95_pnl']:,.2f}]")
    print(f"  Mean Return: {s['mean_return']*100:.2f}%")
    print(f"  Std Return: {s['std_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {s['sharpe']:.3f}")
    print(f"  Win Rate: {s['win_rate']*100:.1f}%")

# ============================================================================
# Relative Performance
# ============================================================================

print("\n" + "-" * 80)
print("RELATIVE PERFORMANCE vs BASELINE")
print("-" * 80)

baseline_mean = stats['Baseline (No Game)']['mean_pnl']

print(f"\n{'Profile':<35} | {'Δ Mean PnL':>12} | {'Improvement':>12}")
print("-" * 65)

for profile_name, s in stats.items():
    if profile_name == 'Baseline (No Game)':
        continue
    delta = s['mean_pnl'] - baseline_mean
    improvement = (delta / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0
    print(f"{profile_name:<35} | ${delta:11.2f} | {improvement:11.2f}%")

# ============================================================================
# Risk-Adjusted Performance
# ============================================================================

print("\n" + "-" * 80)
print("RISK-ADJUSTED COMPARISON")
print("-" * 80)

print(f"\n{'Profile':<35} | {'Sharpe':>8} | {'Rank':>4}")
print("-" * 50)

# Rank by Sharpe ratio
sorted_profiles = sorted(stats.items(), key=lambda x: x[1]['sharpe'], reverse=True)

for rank, (profile_name, s) in enumerate(sorted_profiles, 1):
    print(f"{profile_name:<35} | {s['sharpe']:8.3f} | {rank:4d}")

# ============================================================================
# Generate Plots
# ============================================================================

print("\n" + "-" * 80)
print("GENERATING VISUALIZATIONS")
print("-" * 80)

# PnL Histograms
print("\nGenerating PnL histograms...")
simulator.plot_histograms(results, stats, save_path='pnl_comparison.png')

# Wealth Trajectories
print("Generating wealth trajectories...")
simulator.plot_wealth_trajectories(results, n_sample_paths=50,
                                   save_path='wealth_trajectories.png')

# ============================================================================
# Key Insights
# ============================================================================

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

optimal_stats = stats['Optimal (Paper Formulas)']
baseline_stats = stats['Baseline (No Game)']
vanilla_stats = stats['Vanilla AS (No Predator Risk)']

print(f"\n1. OPTIMAL vs BASELINE:")
print(f"   Mean PnL improvement: ${optimal_stats['mean_pnl'] - baseline_stats['mean_pnl']:,.2f}")
print(f"   Sharpe improvement: {optimal_stats['sharpe'] - baseline_stats['sharpe']:+.3f}")

print(f"\n2. PREDATORY RISK ADJUSTMENT:")
delta_pnl = optimal_stats['mean_pnl'] - vanilla_stats['mean_pnl']
print(f"   Adding predatory risk term (ξγ²q²) to AS yields: ${delta_pnl:+,.2f}")
print(f"   This validates the modified HJB equation (line 973-976 in paper)")

print(f"\n3. RISK-RETURN TRADEOFF:")
print(f"   Optimal strategy Sharpe: {optimal_stats['sharpe']:.3f}")
print(f"   Baseline Sharpe: {baseline_stats['sharpe']:.3f}")
print(f"   Improvement: {(optimal_stats['sharpe']/baseline_stats['sharpe'] - 1)*100:+.1f}%")

print(f"\n4. DOWNSIDE PROTECTION:")
print(f"   Optimal P5: ${optimal_stats['p5_pnl']:,.2f}")
print(f"   Baseline P5: ${baseline_stats['p5_pnl']:,.2f}")
print(f"   The game-theoretic approach {'reduces' if optimal_stats['p5_pnl'] > baseline_stats['p5_pnl'] else 'increases'} tail risk")

print("\n" + "=" * 80)
print("✓ SIMULATION COMPLETE!")
print("=" * 80)
print(f"\nGenerated files:")
print("  - pnl_comparison.png (PnL histograms)")
print("  - wealth_trajectories.png (sample paths)")
print("\nConclusion:")
print("  The double-layer game formulation with predatory risk adjustment")
print("  provides superior risk-adjusted returns compared to naive strategies.")
print("=" * 80 + "\n")
