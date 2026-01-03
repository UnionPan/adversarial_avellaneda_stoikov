"""
Counterfactual Market Making Simulation: Vanilla AS vs Equilibrium AS
======================================================================

This script implements the two-layer market making game with regime switching:
- Inner layer: Market Maker vs Strategic Predator
- Outer layer: Regime switching between stable/volatile states

Compares two strategies:
1. Vanilla AS: Standard Avellaneda-Stoikov (unaware of predator)
2. Equilibrium AS: Uses effective volatility σ_eff² = σ² + ξγ (aware of predator)

Based on the risk isomorphism principle from the double-layer hybrid game framework.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', '..'))

# ============================================================================
# Strategy Functions
# ============================================================================

def vanilla_as_spread(q: float, sigma: float, gamma: float, kappa: float) -> float:
    """
    Vanilla Avellaneda-Stoikov spread (unaware of predator).

    Uses actual volatility σ in the standard AS formula.

    Args:
        q: Current inventory
        sigma: Volatility (regime-dependent)
        gamma: Risk aversion
        kappa: Spread sensitivity

    Returns:
        Optimal half-spread δ*
    """
    # Standard AS formula
    reservation_spread = (gamma * sigma**2) / (2 * kappa)
    inventory_skew = (gamma * sigma**2 * q) / (2 * kappa)
    market_making_premium = np.log(1 + gamma / kappa) / kappa

    delta = reservation_spread + market_making_premium + inventory_skew
    delta = np.clip(delta, 0.0001, 0.06)  # Reasonable bounds

    return delta


def equilibrium_as_spread(q: float, sigma: float, gamma: float, kappa: float, xi: float) -> float:
    """
    Equilibrium AS spread (aware of predator).

    Uses effective volatility σ_eff² = σ² + ξγ to account for predatory risk.
    Based on Risk Isomorphism principle (Remark 1 in paper).

    Args:
        q: Current inventory
        sigma: Volatility (regime-dependent)
        gamma: Risk aversion
        kappa: Spread sensitivity
        xi: Predator cost coefficient

    Returns:
        Optimal half-spread δ*
    """
    # Effective volatility accounting for predator risk
    # σ_eff² = σ² + ξ·γ
    sigma_eff_squared = sigma**2 + xi * gamma
    sigma_eff = np.sqrt(sigma_eff_squared)

    # Use effective volatility in standard AS formula
    reservation_spread = (gamma * sigma_eff_squared) / (2 * kappa)
    inventory_skew = (gamma * sigma_eff_squared * q) / (2 * kappa)
    market_making_premium = np.log(1 + gamma / kappa) / kappa

    delta_eq = reservation_spread + market_making_premium + inventory_skew
    delta_eq = np.clip(delta_eq, 0.0001, 0.06)  # Allow wider spreads

    return delta_eq


def predator_optimal_drift(q: float, gamma: float, xi: float) -> float:
    """
    Predator's optimal drift control from HJB analysis.

    From paper equation (5): w*(q) = -ξ·γ·q

    Manipulates price against MM inventory:
    - If q > 0 (MM long): apply negative drift (push price down)
    - If q < 0 (MM short): apply positive drift (push price up)

    Args:
        q: MM's current inventory
        gamma: MM's risk aversion
        xi: Predator cost coefficient

    Returns:
        Optimal drift w*
    """
    # Paper's closed-form solution
    w_opt = -xi * gamma * q

    # Bound drift to prevent unrealistic price manipulation
    # ±20% annualized drift max
    w_opt = np.clip(w_opt, -0.2, 0.2)

    return w_opt


# ============================================================================
# Calibrated Parameters from BTC Data
# ============================================================================

print("="*80)
print("COUNTERFACTUAL MARKET MAKING SIMULATION")
print("Dec 12, 2025, 15:00 - Dec 13, 03:00 (12 hours)")
print("="*80)

print("\n[1] Loading calibrated parameters...")

# ============================================================================
# Load Calibrated Parameters from CSV files (or use defaults)
# ============================================================================

def load_calibrated_parameters():
    """
    Load calibrated parameters from CSV files produced by calibration scripts.
    Falls back to hardcoded defaults if files don't exist.
    """
    # Default values (fallback)
    params = {
        'SIGMA_STABLE': 0.225271,
        'SIGMA_VOLATILE': 0.530462,
        'BASE_TRANSITION_RATE': 30.0,
        'LAMBDA_0': 250_000,
        'KAPPA': 10.0,
        'AVG_TRADE_SIZE': 0.05,
    }

    sources = {
        'regime': 'default',
        'microstructure': 'default',
    }

    # Try to load regime parameters
    regime_paths = [
        '../results/regime_parameters.csv',
        '../../results/regime_parameters.csv',
    ]

    for path in regime_paths:
        try:
            regime_df = pd.read_csv(path)
            params['SIGMA_STABLE'] = regime_df['SIGMA_STABLE'].iloc[0]
            params['SIGMA_VOLATILE'] = regime_df['SIGMA_VOLATILE'].iloc[0]
            params['BASE_TRANSITION_RATE'] = regime_df['BASE_TRANSITION_RATE'].iloc[0]
            sources['regime'] = path
            print(f"  ✓ Loaded regime parameters from: {path}")
            break
        except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
            continue

    if sources['regime'] == 'default':
        print(f"  ⚠ Using default regime parameters (run btc_regime_calibration.py first)")

    # Try to load microstructure parameters
    micro_paths = [
        '../results/microstructure_parameters.csv',
        '../../results/microstructure_parameters.csv',
    ]

    for path in micro_paths:
        try:
            micro_df = pd.read_csv(path)
            params['LAMBDA_0'] = micro_df['LAMBDA_0'].iloc[0]
            params['KAPPA'] = micro_df['KAPPA'].iloc[0]
            params['AVG_TRADE_SIZE'] = micro_df['AVG_TRADE_SIZE'].iloc[0]
            sources['microstructure'] = path
            print(f"  ✓ Loaded microstructure parameters from: {path}")
            break
        except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
            continue

    if sources['microstructure'] == 'default':
        print(f"  ⚠ Using default microstructure parameters (run gemini_microstructure_calibration.py first)")

    return params, sources

# Load parameters
calibrated_params, param_sources = load_calibrated_parameters()

# Regime parameters
SIGMA_STABLE = calibrated_params['SIGMA_STABLE']
SIGMA_VOLATILE = calibrated_params['SIGMA_VOLATILE']
MU_STABLE = 0.0              # Zero drift - only predator affects drift
MU_VOLATILE = 0.0            # Zero drift - only predator affects drift
BASE_TRANSITION_RATE = calibrated_params['BASE_TRANSITION_RATE']

# Microstructure parameters
LAMBDA_0 = calibrated_params['LAMBDA_0']
KAPPA = calibrated_params['KAPPA']
AVG_TRADE_SIZE = calibrated_params['AVG_TRADE_SIZE']

# Predator parameters and risk aversion
XI = 20.0                    # Predator cost coefficient (ξ·γ = 0.4 for strong predator effect)
GAMMA = 0.02                 # Risk aversion (MM spread calculation - tighter spreads)

# Simulation parameters
DT_SECONDS = 15.0            # 15-second timestep
Q_MAX = 10                   # Max inventory in BTC (±10 BTC)

# Time parameters (12 hours)
HORIZON_HOURS = 12
N_STEPS = int(HORIZON_HOURS * 3600 / DT_SECONDS)
DT_ANNUAL = DT_SECONDS / (365.25 * 24 * 3600)

print(f"\nParameter sources:")
print(f"  Regime parameters: {param_sources['regime']}")
print(f"  Microstructure parameters: {param_sources['microstructure']}")

print(f"\nCalibrated values:")
print(f"  σ_stable: {SIGMA_STABLE:.4f} ({SIGMA_STABLE*100:.2f}%)")
print(f"  σ_volatile: {SIGMA_VOLATILE:.4f} ({SIGMA_VOLATILE*100:.2f}%)")
print(f"  Base transition rate: {BASE_TRANSITION_RATE:.1f} per day")
print(f"  λ₀: {LAMBDA_0:,} per year")
print(f"  κ: {KAPPA}")
print(f"  Avg trade size: {AVG_TRADE_SIZE:.4f} BTC")

print(f"\nSimulation parameters:")
print(f"  Timesteps: {N_STEPS:,} (dt={DT_SECONDS}s)")
print(f"  Horizon: {HORIZON_HOURS} hours")

# Initial conditions (Dec 12, 15:00)
S0 = 90863.90
REGIME_0 = 0  # Start in STABLE regime (afternoon trading typically calmer)
CASH_0 = 0.0
Q_0 = 0

print(f"\nInitial state:")
print(f"  S₀: ${S0:,.2f}")
print(f"  Regime: {'Volatile' if REGIME_0 == 1 else 'Stable'}")

# ============================================================================
# Step 2: Define Strategies
# ============================================================================

print("\n[2] Defining optimal strategies...")
print("  ✓ Vanilla AS spread (no predator)")
print("  ✓ Equilibrium AS spread (with predator consideration)")
print("  ✓ Strategic predatory trader")

# ============================================================================
# Step 3: Build Simulation Engine
# ============================================================================

print("\n[3] Building simulation engine...")

class CounterfactualSimulator:
    """
    Simulates market making with regime switching and strategic predator.

    Both Vanilla AS and Equilibrium AS face the SAME strategic predator
    who observes their inventory and applies optimal adversarial drift.
    """

    def __init__(self, params):
        self.params = params

    def simulate_regime_path(self, n_steps, regime_0, f=0.0, g=0.0):
        """
        Simulate regime switching path using continuous-time Markov chain.

        Transition rate controlled by macro players f (attacker) and g (stabilizer).
        For counterfactual, we set f=g=0 (no macro intervention).
        """
        regimes = np.zeros(n_steps, dtype=int)
        regimes[0] = regime_0

        # Base transition rate (per day)
        mu_01 = BASE_TRANSITION_RATE  # Stable → Volatile
        mu_10 = BASE_TRANSITION_RATE  # Volatile → Stable

        # Convert to per-timestep probability
        p_01 = mu_01 * DT_ANNUAL
        p_10 = mu_10 * DT_ANNUAL

        for i in range(1, n_steps):
            if regimes[i-1] == 0:  # Currently stable
                if np.random.random() < p_01:
                    regimes[i] = 1  # Switch to volatile
                else:
                    regimes[i] = 0  # Stay stable
            else:  # Currently volatile
                if np.random.random() < p_10:
                    regimes[i] = 0  # Switch to stable
                else:
                    regimes[i] = 1  # Stay volatile

        return regimes

    def simulate_order_arrivals(self, delta_bid, delta_ask):
        """
        Simulate Poisson order arrivals for bid and ask sides.

        Intensity: Λ(δ) = λ₀ · exp(-κ·δ)
        """
        # Per-timestep arrival rates
        lambda_bid = LAMBDA_0 * np.exp(-KAPPA * delta_bid) * DT_ANNUAL
        lambda_ask = LAMBDA_0 * np.exp(-KAPPA * delta_ask) * DT_ANNUAL

        # Sample Poisson arrivals
        n_bid = np.random.poisson(lambda_bid)
        n_ask = np.random.poisson(lambda_ask)

        return n_bid, n_ask

    def run_trajectory(self, strategy='vanilla', verbose=False):
        """
        Run a single trajectory of the market making game.

        BOTH strategies face a strategic predator that responds to their inventory.
        Difference: vanilla is unaware and uses standard AS formula,
                   equilibrium is aware and adjusts spreads accordingly.
        """
        # Initialize
        n_steps = self.params['n_steps']

        # Simulate regime path (same for both strategies in same trajectory seed)
        regimes = self.simulate_regime_path(n_steps, self.params['regime_0'], f=0.0, g=0.0)
        sigma_path = np.array([SIGMA_VOLATILE if r == 1 else SIGMA_STABLE for r in regimes])

        # Market maker state
        S = np.zeros(n_steps)
        q = np.zeros(n_steps)
        cash = np.zeros(n_steps)
        pnl = np.zeros(n_steps)

        # Strategy tracking
        spreads = np.zeros(n_steps)
        bids = np.zeros(n_steps)
        asks = np.zeros(n_steps)
        predator_drifts = np.zeros(n_steps)

        # Initial conditions
        S[0] = self.params['S0']
        q[0] = self.params['q0']
        cash[0] = self.params['cash0']
        pnl[0] = cash[0] + q[0] * S[0]

        # INTEGRATED SIMULATION: Predator responds to actual inventory in real-time
        for i in range(1, n_steps):
            regime = regimes[i-1]
            sigma = sigma_path[i-1]

            # Predator observes current inventory and sets drift
            predator_drifts[i-1] = predator_optimal_drift(q[i-1], GAMMA, XI)

            # Simulate price step with predator drift
            mu_base = MU_VOLATILE if regime == 1 else MU_STABLE  # Both zero
            mu_total = mu_base + predator_drifts[i-1]
            dW = np.random.normal(0, np.sqrt(DT_ANNUAL))
            dS = S[i-1] * (mu_total * DT_ANNUAL + sigma * dW)
            S[i] = max(S[i-1] + dS, 1.0)  # Price floor

            # Choose spread strategy
            if strategy == 'vanilla':
                delta_opt = vanilla_as_spread(q[i-1], sigma, GAMMA, KAPPA)
            else:  # equilibrium
                delta_opt = equilibrium_as_spread(q[i-1], sigma, GAMMA, KAPPA, XI)

            spreads[i-1] = delta_opt

            # Post quotes
            bid = S[i-1] * (1 - delta_opt)
            ask = S[i-1] * (1 + delta_opt)
            bids[i-1] = bid
            asks[i-1] = ask

            # Sample arrivals (number of orders)
            n_buy, n_sell = self.simulate_order_arrivals(delta_opt, delta_opt)

            # Update inventory (capped) - each order is AVG_TRADE_SIZE BTC
            # n_buy = customers buy from MM (hit ask) → MM sells → inventory DECREASES
            # n_sell = customers sell to MM (hit bid) → MM buys → inventory INCREASES
            dq = (n_sell - n_buy) * AVG_TRADE_SIZE
            q[i] = np.clip(q[i-1] + dq, -Q_MAX, Q_MAX)

            # Calculate actual executed quantity (respecting inventory limits)
            actual_dq = q[i] - q[i-1]
            actual_buy_qty = max(0, actual_dq) if actual_dq > 0 else 0
            actual_sell_qty = max(0, -actual_dq) if actual_dq < 0 else 0

            # Update cash
            # When MM sells (inventory decreases): receive ASK (higher price)
            # When MM buys (inventory increases): pay BID (lower price)
            cash[i] = cash[i-1] + actual_sell_qty * ask - actual_buy_qty * bid

            # Mark-to-market PnL
            pnl[i] = cash[i] + q[i] * S[i]

        # Terminal liquidation (with penalty)
        liquidation_penalty = 0.001  # 10 bps
        terminal_pnl = cash[-1] + q[-1] * S[-1] * (1 - liquidation_penalty * np.sign(q[-1]))

        return {
            'S': S,
            'q': q,
            'cash': cash,
            'pnl': pnl,
            'terminal_pnl': terminal_pnl,
            'final_inventory': q[-1],
            'spreads': spreads,
            'bids': bids,
            'asks': asks,
            'regimes': regimes,
            'predator_drifts': predator_drifts,
        }

# Create simulator
simulator_params = {
    'n_steps': N_STEPS,
    'S0': S0,
    'q0': Q_0,
    'cash0': CASH_0,
    'regime_0': REGIME_0,
}

simulator = CounterfactualSimulator(simulator_params)
print("  ✓ Simulation engine ready")

# ============================================================================
# Step 4: Run Monte Carlo Simulations
# ============================================================================

print("\n[4] Running Monte Carlo simulations (1000 trajectories)...")

N_TRAJECTORIES = 1000  # Full Monte Carlo simulation

strategies = ['vanilla', 'equilibrium']

results = {}
for strategy in strategies:
    print(f"\n  [{strategy.title()} AS ({'Unaware' if strategy == 'vanilla' else 'Aware'} of Predator)]")

    terminal_pnls = []
    final_inventories = []
    example_trajectory = None

    np.random.seed(42)  # Reproducibility

    for i in range(N_TRAJECTORIES):
        traj = simulator.run_trajectory(strategy=strategy)
        terminal_pnls.append(traj['terminal_pnl'])
        final_inventories.append(traj['final_inventory'])

        # Save first trajectory as example
        if i == 0:
            example_trajectory = traj

        if (i + 1) % 100 == 0:
            print(f"    Progress: {i+1}/{N_TRAJECTORIES}")

    results[strategy] = {
        'terminal_pnls': np.array(terminal_pnls),
        'final_inventories': np.array(final_inventories),
        'example': example_trajectory,
    }

    # Summary statistics
    mean_pnl = np.mean(terminal_pnls)
    std_pnl = np.std(terminal_pnls)
    median_pnl = np.median(terminal_pnls)
    p5 = np.percentile(terminal_pnls, 5)
    p95 = np.percentile(terminal_pnls, 95)

    print(f"    Mean PnL: ${mean_pnl:,.2f}")
    print(f"    Std PnL: ${std_pnl:,.2f}")
    print(f"    5%-95% Range: ${p5:,.2f} to ${p95:,.2f}")

# ============================================================================
# Step 5: Visualization
# ============================================================================

print("\n[5] Creating visualizations...")

# Create multi-panel figure (increased height for better visibility)
fig = plt.figure(figsize=(18, 20))
gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3, height_ratios=[1.5, 1.3, 1.0])

# Use EQUILIBRIUM example trajectory for detailed plot
example = results['equilibrium']['example']
t_hours = np.arange(N_STEPS) * DT_SECONDS / 3600  # Hours from start

# Create actual time labels (Dec 12, 15:00 onwards)
start_time = datetime(2025, 12, 12, 15, 0)
time_labels = [start_time + timedelta(hours=h) for h in t_hours]

# ========== TOP LEFT: Price Evolution with Spread Band ==========
ax1 = fig.add_subplot(gs[0, :])

# Plot mid price with regime coloring
for i in range(len(time_labels) - 1):
    color = 'red' if example['regimes'][i] == 1 else 'green'
    ax1.plot([time_labels[i], time_labels[i+1]], [example['S'][i], example['S'][i+1]],
             color=color, linewidth=2.5, alpha=0.8, zorder=3)

# Add bid-ask spread band (EQUILIBRIUM AS) - 10x for visibility
# Calculate 10x spread for visualization
spread_multiplier = 10
enhanced_bids = example['S'][:-1] * (1 - (1 - example['bids'][:-1]/example['S'][:-1]) * spread_multiplier)
enhanced_asks = example['S'][:-1] * (1 + (example['asks'][:-1]/example['S'][:-1] - 1) * spread_multiplier)
ax1.fill_between(time_labels[:-1], enhanced_bids, enhanced_asks,
                 color='blue', alpha=0.2, label=f'Bid-Ask Spread (×{spread_multiplier} for visibility)', zorder=2)

ax1.set_ylabel('Price ($)', fontsize=13)
ax1.set_title('Price Evolution with Optimal Spread Band (Equilibrium AS vs Strategic Predator)',
              fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, zorder=1)
ax1.set_xlim([time_labels[0], time_labels[-1]])
ax1.legend(fontsize=11, loc='upper left')

# Format x-axis with time labels
ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
fig.autofmt_xdate()

# ========== MIDDLE LEFT: Inventory ==========
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(time_labels, example['q'], color='purple', linewidth=2)
ax2.axhline(0, color='black', linestyle='--', alpha=0.3)
ax2.set_ylabel('Inventory (BTC)', fontsize=13)
ax2.set_title('Inventory Evolution', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

# ========== MIDDLE RIGHT: Spread ==========
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(time_labels[:-1], example['spreads'][:-1] * 10000, color='orange', linewidth=2)
ax3.set_ylabel('Spread (bps)', fontsize=13)
ax3.set_title('Optimal Spread (Equilibrium AS)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

# ========== BOTTOM LEFT: PnL Trajectory ==========
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(time_labels, example['pnl'], color='darkgreen', linewidth=2)
ax4.axhline(0, color='black', linestyle='--', alpha=0.3)
ax4.set_xlabel('Time', fontsize=13)
ax4.set_ylabel('PnL ($)', fontsize=13)
ax4.set_title('Cumulative PnL', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

# ========== BOTTOM RIGHT: Terminal PnL Distributions ==========
ax5 = fig.add_subplot(gs[2, 1])

# Plot distributions
vanilla_pnls = results['vanilla']['terminal_pnls']
equilibrium_pnls = results['equilibrium']['terminal_pnls']

bins = np.linspace(min(vanilla_pnls.min(), equilibrium_pnls.min()),
                   max(vanilla_pnls.max(), equilibrium_pnls.max()), 40)

ax5.hist(vanilla_pnls, bins=bins, alpha=0.5, label='Vanilla AS', color='red', edgecolor='black')
ax5.hist(equilibrium_pnls, bins=bins, alpha=0.5, label='Equilibrium AS', color='blue', edgecolor='black')

# Add mean lines
ax5.axvline(vanilla_pnls.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Vanilla Mean: ${vanilla_pnls.mean():.0f}')
ax5.axvline(equilibrium_pnls.mean(), color='blue', linestyle='--', linewidth=2,
           label=f'Equilibrium Mean: ${equilibrium_pnls.mean():.0f}')

ax5.set_xlabel('Terminal PnL ($)', fontsize=13)
ax5.set_ylabel('Frequency', fontsize=13)
ax5.set_title(f'Terminal PnL Distribution ({N_TRAJECTORIES} paths)', fontsize=13, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

# Get paths
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
results_dir = script_dir.parent / 'results'
results_dir.mkdir(exist_ok=True)

output_path = results_dir / 'counterfactual_simulation_results.png'
plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
print(f"\n✓ Saved visualization: {output_path.relative_to(project_root)}")

# ============================================================================
# Step 6: Export Results
# ============================================================================

print("\n[6] Exporting results...")

# Compute behavioral metrics
def compute_metrics(strategy_results, strategy_name):
    pnls = strategy_results['terminal_pnls']
    example = strategy_results['example']

    # PnL metrics
    mean_pnl = np.mean(pnls)
    median_pnl = np.median(pnls)
    std_pnl = np.std(pnls)
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0
    p5 = np.percentile(pnls, 5)
    p95 = np.percentile(pnls, 95)
    min_pnl = np.min(pnls)
    max_pnl = np.max(pnls)

    # Behavioral metrics
    avg_spread_bps = np.mean(example['spreads']) * 10000
    spread_std_bps = np.std(example['spreads']) * 10000
    avg_abs_inventory = np.mean(np.abs(example['q']))
    max_inventory = np.max(np.abs(example['q']))
    inventory_turnover = np.sum(np.abs(np.diff(example['q'])))
    regime_switches = np.sum(np.diff(example['regimes']) != 0)
    pct_volatile = np.mean(example['regimes'] == 1) * 100
    avg_predator_drift = np.mean(np.abs(example['predator_drifts']))

    return {
        'Strategy': strategy_name,
        'Mean PnL': mean_pnl,
        'Median PnL': median_pnl,
        'Std PnL': std_pnl,
        'Sharpe-like': sharpe,
        '5th Percentile': p5,
        '95th Percentile': p95,
        'Min PnL': min_pnl,
        'Max PnL': max_pnl,
        'Avg Spread (bps)': avg_spread_bps,
        'Spread Std (bps)': spread_std_bps,
        'Avg |Inventory|': avg_abs_inventory,
        'Max |Inventory|': max_inventory,
        'Inventory Turnover': inventory_turnover,
        'Regime Switches': regime_switches,
        '% Time Volatile': pct_volatile,
        'Avg |Predator Drift|': avg_predator_drift,
    }

# Create summary table
summary_data = []
summary_data.append(compute_metrics(results['vanilla'], 'Vanilla AS (Unaware of Predator)'))
summary_data.append(compute_metrics(results['equilibrium'], 'Equilibrium AS (Aware of Predator)'))

summary_df = pd.DataFrame(summary_data)
csv_path = results_dir / 'counterfactual_simulation_summary.csv'
summary_df.to_csv(str(csv_path), index=False)
print(f"✓ Saved summary: {csv_path.relative_to(project_root)}")

# Export example trajectory
example_traj = results['equilibrium']['example']
traj_df = pd.DataFrame({
    'Time (hours)': t_hours,
    'Time': time_labels,
    'Price': example_traj['S'],
    'Inventory': example_traj['q'],
    'Cash': example_traj['cash'],
    'PnL': example_traj['pnl'],
    'Spread (bps)': example_traj['spreads'] * 10000,
    'Bid': example_traj['bids'],
    'Ask': example_traj['asks'],
    'Regime': example_traj['regimes'],
    'Predator Drift': example_traj['predator_drifts'],
})
traj_path = results_dir / 'example_trajectory.csv'
traj_df.to_csv(str(traj_path), index=False)
print(f"✓ Saved example trajectory: {traj_path.relative_to(project_root)}")

# ============================================================================
# Final Summary
# ============================================================================

print("\n" + "="*80)
print("COUNTERFACTUAL SIMULATION COMPLETE")
print("="*80)

print("\nKEY FINDINGS:\n")

for strategy_name in ['Vanilla AS (Unaware of Predator)', 'Equilibrium AS (Aware of Predator)']:
    metrics = summary_df[summary_df['Strategy'] == strategy_name].iloc[0]

    print(f"{strategy_name}:")
    print(f"  Mean PnL: ${metrics['Mean PnL']:.0f}")
    print(f"  Median PnL: ${metrics['Median PnL']:.0f}")
    print(f"  Sharpe-like: {metrics['Sharpe-like']:.3f}")
    print(f"\n  Behavioral Metrics:")
    print(f"    Avg Spread: {metrics['Avg Spread (bps)']:.2f} bps")
    print(f"    Avg |Inventory|: {metrics['Avg |Inventory|']:.2f} BTC")
    print(f"    Max |Inventory|: {metrics['Max |Inventory|']:.2f} BTC")
    print(f"    Regime Switches: {int(metrics['Regime Switches'])}")
    print(f"    % Time Volatile: {metrics['% Time Volatile']:.1f}%")
    print(f"    Avg |Predator Drift|: {metrics['Avg |Predator Drift|']:.4f}")
    print()

# Compute improvement
vanilla_mean = summary_df[summary_df['Strategy'] == 'Vanilla AS (Unaware of Predator)']['Mean PnL'].iloc[0]
equilibrium_mean = summary_df[summary_df['Strategy'] == 'Equilibrium AS (Aware of Predator)']['Mean PnL'].iloc[0]
improvement = (equilibrium_mean / vanilla_mean - 1) * 100

print(f"Improvement (Equilibrium vs Vanilla): +{improvement:.1f}%")

print("\nFILES GENERATED:")
print(f"  - {(results_dir / 'counterfactual_simulation_results.png').relative_to(project_root)}")
print(f"  - {(results_dir / 'counterfactual_simulation_summary.csv').relative_to(project_root)}")
print(f"  - {(results_dir / 'example_trajectory.csv').relative_to(project_root)}")

print("\n" + "="*80)
