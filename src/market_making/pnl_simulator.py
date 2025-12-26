"""
Multi-Path PnL Simulator for Market Making Game

Simulates multiple episodes with different seeds and generates:
- PnL histograms comparing strategies
- Summary statistics (mean, std, Sharpe, percentiles)
- Wealth trajectory comparisons

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
import warnings

from .market_making_env import make_market_making_env, MarketMakingEnv
from .strategies import StrategyProfile


@dataclass
class SimulationResult:
    """Results from a single simulation run."""
    profile_name: str
    seed: int
    final_wealth: float
    pnl: float
    initial_wealth: float
    final_inventory: float
    total_trades: int
    wealth_path: np.ndarray
    inventory_path: np.ndarray
    regime_stable_pct: float


class PnLSimulator:
    """
    Multi-path PnL simulator for strategy comparison.

    Usage:
        simulator = PnLSimulator(
            env_config={'S_0': 50000, 'gamma': 0.01, 'xi': 0.02},
            n_paths=1000,
        )

        profiles = [
            get_optimal_profile(),
            get_baseline_profile(),
            get_mm_only_profile(),
        ]

        results = simulator.run(profiles)
        simulator.plot_histograms(results)
        simulator.print_statistics(results)
    """

    def __init__(
        self,
        env_config: Optional[Dict] = None,
        n_paths: int = 1000,
        verbose: bool = True,
    ):
        """
        Initialize PnL simulator.

        Args:
            env_config: Configuration dict for make_market_making_env()
            n_paths: Number of Monte Carlo paths per profile
            verbose: Print progress
        """
        self.env_config = env_config or {}
        self.n_paths = n_paths
        self.verbose = verbose

    def run_single_path(
        self,
        profile: StrategyProfile,
        seed: int,
        env: Optional[MarketMakingEnv] = None,
    ) -> SimulationResult:
        """
        Run single episode with given profile and seed.

        Args:
            profile: Strategy profile to use
            seed: Random seed
            env: Pre-configured environment (if None, creates new)

        Returns:
            SimulationResult with episode statistics
        """
        if env is None:
            env = make_market_making_env(**self.env_config)

        obs, info = env.reset(seed=seed)
        initial_wealth = obs['wealth'][0]

        wealth_path = [initial_wealth]
        inventory_path = [obs['inventory'][0]]

        for t in range(env.max_steps):
            # Get actions from profile
            delta_bid, delta_ask = profile.mm(obs, regime=info.get('regime'))
            w_t = profile.predator(obs)

            # Macro updates at macro frequency
            if t % env.macro_freq == 0:
                f_t, g_t = profile.macro(obs)
            else:
                f_t, g_t = None, None

            # Step
            action = {
                'delta_bid': np.array([delta_bid]),
                'delta_ask': np.array([delta_ask])
            }
            obs, reward, terminated, truncated, info = env.step(action, w_t, f_t, g_t)

            wealth_path.append(obs['wealth'][0])
            inventory_path.append(obs['inventory'][0])

            if terminated or truncated:
                break

        # Extract statistics
        final_wealth = obs['wealth'][0]
        pnl = final_wealth - initial_wealth
        final_inventory = obs['inventory'][0]
        total_trades = sum(env.history['n_buy']) + sum(env.history['n_sell'])
        regime_stable_pct = np.mean(np.array(env.history['regime']) == 0)

        return SimulationResult(
            profile_name=profile.name,
            seed=seed,
            final_wealth=final_wealth,
            pnl=pnl,
            initial_wealth=initial_wealth,
            final_inventory=final_inventory,
            total_trades=total_trades,
            wealth_path=np.array(wealth_path),
            inventory_path=np.array(inventory_path),
            regime_stable_pct=regime_stable_pct,
        )

    def run(
        self,
        profiles: List[StrategyProfile],
        seeds: Optional[List[int]] = None,
    ) -> Dict[str, List[SimulationResult]]:
        """
        Run Monte Carlo simulation for all profiles.

        Args:
            profiles: List of strategy profiles to compare
            seeds: List of random seeds (if None, uses range(n_paths))

        Returns:
            Dict mapping profile name to list of SimulationResults
        """
        if seeds is None:
            seeds = list(range(self.n_paths))
        else:
            self.n_paths = len(seeds)

        results = {profile.name: [] for profile in profiles}

        # Create single environment to share across profiles (same seeds)
        env = make_market_making_env(**self.env_config)

        for i, seed in enumerate(seeds):
            if self.verbose and (i % 100 == 0 or i == self.n_paths - 1):
                print(f"  Simulating path {i+1}/{self.n_paths}...")

            for profile in profiles:
                result = self.run_single_path(profile, seed, env)
                results[profile.name].append(result)

        return results

    def compute_statistics(
        self,
        results: Dict[str, List[SimulationResult]]
    ) -> Dict[str, Dict]:
        """
        Compute summary statistics for each profile.

        Args:
            results: Dict from run()

        Returns:
            Dict mapping profile name to statistics dict
        """
        stats = {}

        for profile_name, profile_results in results.items():
            pnls = np.array([r.pnl for r in profile_results])
            final_wealths = np.array([r.final_wealth for r in profile_results])
            returns = pnls / profile_results[0].initial_wealth

            # Risk-adjusted metrics
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = mean_return / std_return if std_return > 0 else 0

            stats[profile_name] = {
                'mean_pnl': np.mean(pnls),
                'std_pnl': np.std(pnls),
                'median_pnl': np.median(pnls),
                'min_pnl': np.min(pnls),
                'max_pnl': np.max(pnls),
                'p5_pnl': np.percentile(pnls, 5),
                'p95_pnl': np.percentile(pnls, 95),
                'mean_return': mean_return,
                'std_return': std_return,
                'sharpe': sharpe,
                'win_rate': np.mean(pnls > 0),
                'mean_final_wealth': np.mean(final_wealths),
            }

        return stats

    def print_statistics(
        self,
        results: Dict[str, List[SimulationResult]],
        stats: Optional[Dict] = None,
    ):
        """
        Print formatted statistics table.

        Args:
            results: Dict from run()
            stats: Pre-computed statistics (if None, computes)
        """
        if stats is None:
            stats = self.compute_statistics(results)

        print("\n" + "=" * 100)
        print(f"STRATEGY COMPARISON ({self.n_paths} paths)")
        print("=" * 100)
        print(f"{'Profile':<30} | {'Mean PnL':>12} | {'Std PnL':>10} | "
              f"{'Sharpe':>8} | {'Win %':>7} | {'P5':>10} | {'P95':>10}")
        print("-" * 100)

        for profile_name, s in stats.items():
            print(f"{profile_name:<30} | ${s['mean_pnl']:11.2f} | "
                  f"${s['std_pnl']:9.2f} | {s['sharpe']:8.3f} | "
                  f"{s['win_rate']*100:6.1f}% | ${s['p5_pnl']:9.2f} | "
                  f"${s['p95_pnl']:9.2f}")

        print("=" * 100)

    def plot_histograms(
        self,
        results: Dict[str, List[SimulationResult]],
        stats: Optional[Dict] = None,
        save_path: Optional[str] = None,
    ):
        """
        Plot PnL histogram comparison.

        Args:
            results: Dict from run()
            stats: Pre-computed statistics
            save_path: Path to save figure (if None, just displays)
        """
        if stats is None:
            stats = self.compute_statistics(results)

        n_profiles = len(results)
        fig, axes = plt.subplots(n_profiles, 1, figsize=(12, 4*n_profiles))

        if n_profiles == 1:
            axes = [axes]

        colors = plt.cm.Set2(np.linspace(0, 1, n_profiles))

        for idx, (profile_name, profile_results) in enumerate(results.items()):
            ax = axes[idx]
            pnls = np.array([r.pnl for r in profile_results])

            # Histogram
            ax.hist(pnls, bins=50, alpha=0.7, color=colors[idx], edgecolor='black')

            # Vertical lines for statistics
            s = stats[profile_name]
            ax.axvline(s['mean_pnl'], color='red', linestyle='--', linewidth=2,
                      label=f"Mean: ${s['mean_pnl']:.2f}")
            ax.axvline(s['median_pnl'], color='orange', linestyle='--', linewidth=2,
                      label=f"Median: ${s['median_pnl']:.2f}")
            ax.axvline(s['p5_pnl'], color='blue', linestyle=':', linewidth=1.5,
                      label=f"P5: ${s['p5_pnl']:.2f}")
            ax.axvline(s['p95_pnl'], color='blue', linestyle=':', linewidth=1.5,
                      label=f"P95: ${s['p95_pnl']:.2f}")
            ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

            # Labels
            ax.set_xlabel('PnL ($)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{profile_name} | Sharpe: {s["sharpe"]:.3f} | Win: {s["win_rate"]*100:.1f}%',
                        fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'PnL Distribution Comparison ({self.n_paths} paths)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nSaved histogram: {save_path}")

        return fig

    def plot_wealth_trajectories(
        self,
        results: Dict[str, List[SimulationResult]],
        n_sample_paths: int = 20,
        save_path: Optional[str] = None,
    ):
        """
        Plot sample wealth trajectories for each profile.

        Args:
            results: Dict from run()
            n_sample_paths: Number of sample paths to plot per profile
            save_path: Path to save figure
        """
        n_profiles = len(results)
        fig, axes = plt.subplots(1, n_profiles, figsize=(6*n_profiles, 5))

        if n_profiles == 1:
            axes = [axes]

        colors = plt.cm.Set2(np.linspace(0, 1, n_profiles))

        for idx, (profile_name, profile_results) in enumerate(results.items()):
            ax = axes[idx]

            # Sample random paths
            sample_indices = np.random.choice(len(profile_results),
                                             min(n_sample_paths, len(profile_results)),
                                             replace=False)

            # Plot individual paths
            for i in sample_indices:
                wealth_path = profile_results[i].wealth_path
                ax.plot(wealth_path, alpha=0.3, linewidth=0.5, color=colors[idx])

            # Plot mean path
            all_paths = [r.wealth_path for r in profile_results]
            min_len = min(len(p) for p in all_paths)
            aligned_paths = np.array([p[:min_len] for p in all_paths])
            mean_path = np.mean(aligned_paths, axis=0)

            ax.plot(mean_path, color=colors[idx], linewidth=2.5, label='Mean',
                   linestyle='--')
            ax.axhline(profile_results[0].initial_wealth, color='k',
                      linestyle=':', alpha=0.5, label='Initial')

            ax.set_xlabel('Step')
            ax.set_ylabel('Wealth ($)')
            ax.set_title(profile_name, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Wealth Trajectories ({n_sample_paths} sample paths)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved trajectories: {save_path}")

        return fig


def quick_comparison(
    profiles: List[StrategyProfile],
    n_paths: int = 1000,
    env_config: Optional[Dict] = None,
) -> Dict[str, List[SimulationResult]]:
    """
    Quick comparison helper function.

    Args:
        profiles: List of strategy profiles
        n_paths: Number of paths
        env_config: Environment configuration

    Returns:
        Results dict

    Example:
        from market_making.strategies import get_optimal_profile, get_baseline_profile
        from market_making.pnl_simulator import quick_comparison

        results = quick_comparison(
            profiles=[get_optimal_profile(), get_baseline_profile()],
            n_paths=1000,
        )
    """
    simulator = PnLSimulator(env_config=env_config, n_paths=n_paths, verbose=True)

    print(f"\nRunning {n_paths} paths for {len(profiles)} profiles...")
    results = simulator.run(profiles)

    print("\nComputing statistics...")
    stats = simulator.compute_statistics(results)
    simulator.print_statistics(results, stats)

    print("\nGenerating plots...")
    simulator.plot_histograms(results, stats, save_path='pnl_comparison.png')
    simulator.plot_wealth_trajectories(results, save_path='wealth_trajectories.png')

    return results
