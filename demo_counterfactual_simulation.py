"""
Counterfactual Market Making Simulation: Dec 12, 2025

Simulates market making game using calibrated parameters:
- Regime parameters: From Kraken 30-min data
- Microstructure: From Gemini tick data (1/1000 market share)
- Time period: Dec 12, 15:00 - Dec 13, 03:00 (12 hours)
- Trajectories: 100 Monte Carlo paths
- Strategies: Optimal AS spread + Macro attack/defense

Author: Yunian Pan
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Tuple, List

print("=" * 80)
print("COUNTERFACTUAL MARKET MAKING SIMULATION")
print("Dec 12, 2025, 15:00 - Dec 13, 03:00 (12 hours)")
print("=" * 80)

# ============================================================================
# Step 1: Load Calibrated Parameters
# ============================================================================

print("\n[1] Loading calibrated parameters...")

# Regime parameters (from Kraken 30-min calibration)
SIGMA_STABLE = 0.225271      # 22.53% annualized
SIGMA_VOLATILE = 0.530462    # 53.05% annualized
MU_STABLE = 0.0              # Zero drift - only predator affects drift
MU_VOLATILE = 0.0            # Zero drift - only predator affects drift
BASE_TRANSITION_RATE = 30.0  # Per day (avg 48 min holding time) - ensures 5-10 switches over 12h

# Microstructure (from Gemini tick calibration, scaled for 1/1000 market share)
LAMBDA_0 = 250_000           # Base arrival rate (~42 trades/hour for individual MM)
KAPPA = 10.0                 # Spread sensitivity (exp(-κδ) order flow) - higher = more sensitive
AVG_TRADE_SIZE = 0.05        # Average trade size in BTC

# Predator parameters and risk aversion
XI = 20.0                    # Predator cost coefficient (ξ·γ = 0.4 for strong predator effect)
GAMMA = 0.02                 # Risk aversion (MM spread calculation - tighter spreads)

# Simulation parameters
DT_SECONDS = 15.0            # 15-second timestep
Q_MAX = 10                   # Max inventory in BTC (±10 BTC)
LIQUIDATION_PENALTY = 0.0005 # 5 bps for forced liquidation

# Time horizon
T_HOURS = 12
N_STEPS = int(T_HOURS * 3600 / DT_SECONDS)  # 2,880 timesteps
DT_ANNUAL = DT_SECONDS / (252 * 24 * 60 * 60)  # Convert to years

print(f"  σ_stable: {SIGMA_STABLE:.4f} ({SIGMA_STABLE:.2%})")
print(f"  σ_volatile: {SIGMA_VOLATILE:.4f} ({SIGMA_VOLATILE:.2%})")
print(f"  λ₀: {LAMBDA_0:,} per year")
print(f"  κ: {KAPPA}")
print(f"  Timesteps: {N_STEPS:,} (dt={DT_SECONDS}s)")

# Initial conditions (Dec 12, 15:00)
S0 = 90863.90
REGIME_0 = 0  # Start in STABLE regime (afternoon trading typically calmer)
CASH_0 = 0.0
Q_0 = 0

print(f"\nInitial state:")
print(f"  S₀: ${S0:,.2f}")
print(f"  Regime: {'Volatile' if REGIME_0 == 1 else 'Stable'}")

# ============================================================================
# Step 2: Define Optimal Strategies
# ============================================================================

print("\n[2] Defining optimal strategies...")

def vanilla_as_spread(q: float, sigma: float, gamma: float, kappa: float) -> float:
    """
    Vanilla Avellaneda-Stoikov optimal half-spread (no predator consideration)

    δ* = γσ²/2κ + (1/κ)ln(1 + γ/κ) + γσ²q/2κ

    Args:
        q: Current inventory
        sigma: Current volatility (annualized)
        gamma: Risk aversion
        kappa: Spread sensitivity

    Returns:
        Optimal half-spread (fraction of price)
    """
    reservation_spread = (gamma * sigma**2) / (2 * kappa)
    inventory_skew = (gamma * sigma**2 * q) / (2 * kappa)
    market_making_premium = np.log(1 + gamma / kappa) / kappa

    delta_opt = reservation_spread + market_making_premium + inventory_skew
    delta_opt = np.clip(delta_opt, 0.0001, 0.05)  # Allow wider spreads for κ=8

    return delta_opt

def equilibrium_as_spread(q: float, sigma: float, gamma: float, kappa: float, xi: float) -> float:
    """
    Equilibrium AS spread accounting for strategic predatory trader

    From the paper's HJB analysis:
    - Predator creates drift w*(q) = -ξ·γ·q
    - This adds effective volatility to the system
    - Equilibrium spread uses σ_eff² = σ² + ξ·γ instead of σ²

    The predator risk acts like additional volatility, so MM uses
    effective volatility in the standard AS formula.

    Args:
        q: Current inventory
        sigma: Current volatility (annualized)
        gamma: Risk aversion
        kappa: Spread sensitivity
        xi: Predator cost coefficient

    Returns:
        Equilibrium half-spread (fraction of price)
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
    Predator's optimal drift control

    From the paper's HJB analysis, the predator's best response is:
    w*(q) = -ξ·γ·q

    Manipulates price against MM inventory:
    - If q > 0 (MM long), apply negative drift (push price down)
    - If q < 0 (MM short), apply positive drift (push price up)

    Cost-benefit tradeoff: max E[profit from manipulation] - (1/2)w²
    Optimal control: w* = -ξ·γ·q

    Args:
        q: MM inventory (BTC)
        gamma: MM risk aversion parameter
        xi: Predator cost coefficient

    Returns:
        Optimal drift (annualized, bounded)
    """
    # Paper's closed-form solution
    w_opt = -xi * gamma * q

    # Bound drift to prevent unrealistic price manipulation
    # ±20% annualized drift max
    w_opt = np.clip(w_opt, -0.2, 0.2)

    return w_opt

print("  ✓ Vanilla AS spread (no predator)")
print("  ✓ Equilibrium AS spread (with predator consideration)")
print("  ✓ Strategic predatory trader")

# ============================================================================
# Step 3: Simulation Engine
# ============================================================================

print("\n[3] Building simulation engine...")

class MarketMakingSimulator:
    """Monte Carlo simulator for market making game"""

    def __init__(self, params: dict):
        self.params = params

    def simulate_regime_path(self, n_steps: int, initial_regime: int,
                            f: float, g: float) -> np.ndarray:
        """
        Simulate regime switching with macro controls

        Transition rate: μ_ij(f,g) = μ₀_ij + f·λ_att - g·λ_stab

        Args:
            f: Attack control (destabilize) in [0,1]
            g: Defense control (stabilize) in [0,1]
        """
        # Bound controls to [0,1]
        f = np.clip(f, 0.0, 1.0)
        g = np.clip(g, 0.0, 1.0)

        # Base transition matrix (per day)
        base_rate = self.params['base_transition_rate']

        # Convert to per-timestep rate
        dt_daily = DT_SECONDS / (24 * 60 * 60)
        exit_rate_0 = base_rate * dt_daily  # Stable -> Volatile
        exit_rate_1 = base_rate * dt_daily  # Volatile -> Stable

        # Apply macro controls
        # f increases transitions to volatile, g decreases them
        exit_rate_0 = max(0, exit_rate_0 * (1 + f - g))
        exit_rate_1 = max(0, exit_rate_1 * (1 - f + g))

        regimes = np.zeros(n_steps, dtype=int)
        regimes[0] = initial_regime

        for i in range(1, n_steps):
            current_regime = regimes[i-1]
            exit_rate = exit_rate_1 if current_regime == 1 else exit_rate_0

            # Probability of switching in this timestep
            switch_prob = 1 - np.exp(-exit_rate)

            if np.random.rand() < switch_prob:
                regimes[i] = 1 - current_regime  # Switch
            else:
                regimes[i] = current_regime  # Stay

        return regimes

    def simulate_price_path(self, n_steps: int, regimes: np.ndarray,
                           predator_drifts: np.ndarray = None) -> np.ndarray:
        """
        Simulate price evolution with regime-switching GBM + predatory drift

        Args:
            n_steps: Number of timesteps
            regimes: Regime path array
            predator_drifts: Optional predator drift at each timestep (annualized)

        Returns:
            Price path
        """
        S = np.zeros(n_steps)
        S[0] = self.params['S0']

        if predator_drifts is None:
            predator_drifts = np.zeros(n_steps)

        for i in range(1, n_steps):
            regime = regimes[i-1]
            mu_base = MU_VOLATILE if regime == 1 else MU_STABLE
            sigma = SIGMA_VOLATILE if regime == 1 else SIGMA_STABLE

            # Total drift = base regime drift + predator manipulation
            mu_total = mu_base + predator_drifts[i-1]

            # GBM step
            dW = np.random.normal(0, np.sqrt(DT_ANNUAL))
            dS = S[i-1] * (mu_total * DT_ANNUAL + sigma * dW)
            S[i] = max(S[i-1] + dS, 1.0)  # Price floor at $1

        return S

    def simulate_order_arrivals(self, delta_bid: float, delta_ask: float) -> Tuple[int, int]:
        """
        Simulate Poisson order arrivals

        λ(δ) = λ₀ · exp(-κδ)
        """
        lambda_bid = LAMBDA_0 * np.exp(-KAPPA * delta_bid)
        lambda_ask = LAMBDA_0 * np.exp(-KAPPA * delta_ask)

        # Expected arrivals in this timestep
        intensity_bid = lambda_bid * DT_ANNUAL
        intensity_ask = lambda_ask * DT_ANNUAL

        # Sample from Poisson
        n_buy = np.random.poisson(intensity_ask)  # Buys hit our ask
        n_sell = np.random.poisson(intensity_bid)  # Sells hit our bid

        return n_buy, n_sell

    def run_trajectory(self, strategy: str = 'vanilla', verbose: bool = False) -> dict:
        """
        Run single trajectory with given MM strategy

        BOTH strategies face a strategic predator that responds to their inventory.
        Difference: vanilla is unaware and uses standard AS formula,
                   equilibrium is aware and adjusts spreads accordingly.

        Args:
            strategy: 'vanilla' (unaware) or 'equilibrium' (aware)
            verbose: Print debug info

        Returns:
            dict with arrays: S, regimes, spreads, q, cash, pnl
        """
        # Initialize
        n_steps = self.params['n_steps']

        # Simulate regime path (no macro intervention, natural transitions)
        regimes = self.simulate_regime_path(n_steps, self.params['regime_0'], f=0.0, g=0.0)
        sigma_path = np.array([SIGMA_VOLATILE if r == 1 else SIGMA_STABLE for r in regimes])

        # Market maker state
        S = np.zeros(n_steps)
        q = np.zeros(n_steps)
        cash = np.zeros(n_steps)
        pnl = np.zeros(n_steps)
        spreads = np.zeros(n_steps)
        predator_drifts = np.zeros(n_steps)
        bids = np.zeros(n_steps)
        asks = np.zeros(n_steps)

        S[0] = self.params['S0']
        q[0] = self.params['q0']
        cash[0] = self.params['cash0']

        # INTEGRATED SIMULATION: Predator responds to actual inventory in real-time
        for i in range(1, n_steps):
            # Current state
            regime = regimes[i-1]
            sigma = sigma_path[i-1]

            # Predator observes current inventory and sets drift
            predator_drifts[i-1] = predator_optimal_drift(q[i-1], GAMMA, XI)

            # Simulate price step with predator drift
            mu_base = MU_VOLATILE if regime == 1 else MU_STABLE
            mu_total = mu_base + predator_drifts[i-1]
            dW = np.random.normal(0, np.sqrt(DT_ANNUAL))
            dS = S[i-1] * (mu_total * DT_ANNUAL + sigma * dW)
            S[i] = max(S[i-1] + dS, 1.0)  # Price floor

            # Choose spread strategy
            if strategy == 'vanilla':
                # Vanilla AS: no predator awareness
                delta_opt = vanilla_as_spread(q[i-1], sigma, GAMMA, KAPPA)
            else:  # equilibrium
                # Equilibrium AS: aware of predator
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
        terminal_value = cash[-1] + q[-1] * S[-1] * (1 - LIQUIDATION_PENALTY * abs(q[-1]))

        return {
            'S': S,
            'regimes': regimes,
            'spreads': spreads,
            'q': q,
            'cash': cash,
            'pnl': pnl,
            'terminal_pnl': terminal_value,
            'final_inventory': q[-1],
            'predator_drifts': predator_drifts,
            'bids': bids,
            'asks': asks,
        }

# Initialize simulator
sim_params = {
    'S0': S0,
    'regime_0': REGIME_0,
    'q0': Q_0,
    'cash0': CASH_0,
    'n_steps': N_STEPS,
    'base_transition_rate': BASE_TRANSITION_RATE,
}

simulator = MarketMakingSimulator(sim_params)

print("  ✓ Simulation engine ready")

# ============================================================================
# Step 4: Run Monte Carlo Simulations
# ============================================================================

print("\n[4] Running Monte Carlo simulations (1000 trajectories)...")

N_TRAJECTORIES = 1000  # Full Monte Carlo simulation

strategies = ['vanilla', 'equilibrium']

results = {}

for strategy in strategies:
    if strategy == 'vanilla':
        strategy_label = "Vanilla AS (Unaware of Predator)"
    else:
        strategy_label = "Equilibrium AS (Aware of Predator)"
    print(f"\n  [{strategy_label}]")

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
        'label': strategy_label,
    }

    # Summary statistics
    mean_pnl = np.mean(terminal_pnls)
    std_pnl = np.std(terminal_pnls)
    pnl_05 = np.percentile(terminal_pnls, 5)
    pnl_95 = np.percentile(terminal_pnls, 95)

    print(f"    Mean PnL: ${mean_pnl:,.2f}")
    print(f"    Std PnL: ${std_pnl:,.2f}")
    print(f"    5%-95% Range: ${pnl_05:,.2f} to ${pnl_95:,.2f}")

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

# Convert to actual time of day (starting at 15:00)
from datetime import datetime, timedelta
start_time = datetime(2025, 12, 12, 15, 0, 0)  # Dec 12, 15:00
time_labels = [start_time + timedelta(hours=h) for h in t_hours]

# Panel 1: Price evolution with regime coloring + spread band
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

# Format x-axis to show time of day
import matplotlib.dates as mdates
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)

# Add regime legend (no mid price)
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='blue', alpha=0.2, label='Bid-Ask Spread'),
    Patch(facecolor='green', alpha=0.7, label='Stable Regime'),
    Patch(facecolor='red', alpha=0.7, label='Volatile Regime'),
]
ax1.legend(handles=legend_elements, loc='upper right', fontsize=11)

# Panel 2: Spread evolution (dual y-axis with inventory)
ax2 = fig.add_subplot(gs[1, :])
spreads_bps = example['spreads'] * 10000  # Convert to bps
ax2.plot(time_labels, spreads_bps, color='blue', linewidth=2, label='Optimal Spread (Equilibrium AS)')
ax2.set_ylabel('Spread (bps)', fontsize=13, color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.set_xlabel('Time of Day', fontsize=13)
ax2.set_title('Optimal Spread Strategy & Inventory (Equilibrium AS)', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([time_labels[0], time_labels[-1]])

# Format x-axis
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

# Overlay inventory on second y-axis
ax2b = ax2.twinx()
ax2b.plot(time_labels, example['q'], color='orange', linewidth=2, linestyle='--', label='Inventory')
ax2b.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax2b.set_ylabel('Inventory (BTC)', fontsize=13, color='orange')
ax2b.tick_params(axis='y', labelcolor='orange')

# Combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

# Panel 3: PnL distribution comparison
ax3 = fig.add_subplot(gs[2, 0])
colors = {'vanilla': 'blue', 'equilibrium': 'orange'}
for strategy, res in results.items():
    ax3.hist(res['terminal_pnls'], bins=40, alpha=0.6, label=res['label'],
             color=colors[strategy], edgecolor='black', linewidth=0.5)

ax3.set_xlabel('Terminal PnL ($)', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title(f'Terminal PnL Distribution ({N_TRAJECTORIES} Trajectories)', fontsize=13)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')
ax3.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

# Panel 4: Summary statistics
ax4 = fig.add_subplot(gs[2, 1])
ax4.axis('off')

summary_text = "STRATEGY COMPARISON\n" + "=" * 50 + "\n\n"

for strategy, res in results.items():
    pnls = res['terminal_pnls']
    summary_text += f"{res['label']}:\n"
    summary_text += f"  Mean:    ${np.mean(pnls):>10,.0f}\n"
    summary_text += f"  Median:  ${np.median(pnls):>10,.0f}\n"
    summary_text += f"  Std:     ${np.std(pnls):>10,.0f}\n"
    summary_text += f"  Sharpe:  {np.mean(pnls)/np.std(pnls):>10.3f}\n"
    summary_text += f"  5%-95%:  ${np.percentile(pnls, 5):>8,.0f}\n"
    summary_text += f"           ${np.percentile(pnls, 95):>8,.0f}\n\n"

# Compare strategies
vanilla_mean = np.mean(results['vanilla']['terminal_pnls'])
eq_mean = np.mean(results['equilibrium']['terminal_pnls'])
improvement = ((eq_mean - vanilla_mean) / abs(vanilla_mean)) * 100

summary_text += f"\nIMPROVEMENT:\n"
summary_text += f"  Equilibrium vs Vanilla:\n"
summary_text += f"  {improvement:+.1f}% mean PnL change\n\n"

summary_text += "PARAMETERS:\n"
summary_text += f"  Period: Dec 12, 15:00-03:00\n"
summary_text += f"  Timestep: {DT_SECONDS}s ({N_STEPS:,})\n"
summary_text += f"  Trajectories: {N_TRAJECTORIES}\n"
summary_text += f"  γ: {GAMMA}, κ: {KAPPA}, ξ: {XI}\n"
summary_text += f"  Regime rate: {BASE_TRANSITION_RATE:.1f}/day\n"

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
         fontsize=9, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Market Making Strategy Comparison: Vanilla AS vs Equilibrium AS (Both Against Strategic Predator)',
             fontsize=15, fontweight='bold', y=0.997)

output_path = 'counterfactual_simulation_results.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved visualization: {output_path}")

# ============================================================================
# Step 6: Export Results
# ============================================================================

print("\n[6] Exporting results...")

# Create summary dataframe with behavioral metrics
summary_data = []
for strategy, res in results.items():
    pnls = res['terminal_pnls']
    inventories = res['final_inventories']
    ex = res['example']  # Example trajectory for behavior analysis

    # Behavioral metrics
    avg_spread_bps = np.mean(ex['spreads']) * 10000
    spread_std_bps = np.std(ex['spreads']) * 10000
    avg_abs_inventory = np.mean(np.abs(ex['q']))
    max_inventory = np.max(np.abs(ex['q']))
    inventory_turnover = np.sum(np.abs(np.diff(ex['q'])))
    regime_switches = np.sum(np.diff(ex['regimes']) != 0)
    pct_volatile = np.mean(ex['regimes'] == 1) * 100
    avg_predator_drift = np.mean(np.abs(ex['predator_drifts']))

    summary_data.append({
        'Strategy': res['label'],
        'Mean PnL': np.mean(pnls),
        'Median PnL': np.median(pnls),
        'Std PnL': np.std(pnls),
        'Sharpe-like': np.mean(pnls) / np.std(pnls),
        '5th Percentile': np.percentile(pnls, 5),
        '95th Percentile': np.percentile(pnls, 95),
        'Min PnL': pnls.min(),
        'Max PnL': pnls.max(),
        'Avg Spread (bps)': avg_spread_bps,
        'Spread Std (bps)': spread_std_bps,
        'Avg |Inventory|': avg_abs_inventory,
        'Max |Inventory|': max_inventory,
        'Inventory Turnover': inventory_turnover,
        'Regime Switches': regime_switches,
        '% Time Volatile': pct_volatile,
        'Avg |Predator Drift|': avg_predator_drift,
    })

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv('counterfactual_simulation_summary.csv', index=False)
print(f"✓ Saved summary: counterfactual_simulation_summary.csv")

# Save example trajectory
example_df = pd.DataFrame({
    'Time (hours)': t_hours,
    'Price': example['S'],
    'Regime': example['regimes'],
    'Spread (bps)': example['spreads'] * 10000,
    'Inventory': example['q'],
    'Cash': example['cash'],
    'PnL': example['pnl'],
})
example_df.to_csv('example_trajectory.csv', index=False)
print(f"✓ Saved example trajectory: example_trajectory.csv")

# ============================================================================
# Final Summary
# ============================================================================

print("\n" + "=" * 80)
print("COUNTERFACTUAL SIMULATION COMPLETE")
print("=" * 80)

print("\nKEY FINDINGS:\n")
for strategy, res in results.items():
    pnls = res['terminal_pnls']
    ex = res['example']

    print(f"{res['label']}:")
    print(f"  Mean PnL: ${np.mean(pnls):,.0f}")
    print(f"  Median PnL: ${np.median(pnls):,.0f}")
    print(f"  Sharpe-like: {np.mean(pnls) / np.std(pnls):.3f}")

    # Behavioral metrics
    print(f"\n  Behavioral Metrics:")
    print(f"    Avg Spread: {np.mean(ex['spreads']) * 10000:.2f} bps")
    print(f"    Avg |Inventory|: {np.mean(np.abs(ex['q'])):.2f} BTC")
    print(f"    Max |Inventory|: {np.max(np.abs(ex['q'])):.2f} BTC")
    print(f"    Regime Switches: {np.sum(np.diff(ex['regimes']) != 0)}")
    print(f"    % Time Volatile: {np.mean(ex['regimes'] == 1) * 100:.1f}%")
    print(f"    Avg |Predator Drift|: {np.mean(np.abs(ex['predator_drifts'])):.4f}\n")

# Show improvement
vanilla_mean = np.mean(results['vanilla']['terminal_pnls'])
eq_mean = np.mean(results['equilibrium']['terminal_pnls'])
improvement_pct = ((eq_mean - vanilla_mean) / abs(vanilla_mean)) * 100
print(f"Improvement (Equilibrium vs Vanilla): {improvement_pct:+.1f}%")

print("\nFILES GENERATED:")
print(f"  - {output_path}")
print(f"  - counterfactual_simulation_summary.csv")
print(f"  - example_trajectory.csv")

print("\n" + "=" * 80)
