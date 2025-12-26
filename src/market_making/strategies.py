"""
Market Making Game Strategies

Implements closed-form strategies from double-layer AS game:
- Predator: Mean-averting drift control
- Market Maker: Avellaneda-Stoikov with predatory risk adjustment
- Macro players: Proportional feedback on stability gap


Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import numpy as np
from typing import Dict, Tuple, Optional
from abc import ABC, abstractmethod


# ============================================================================
# Base Strategy Classes
# ============================================================================

class Strategy(ABC):
    """Base class for all strategies."""

    @abstractmethod
    def __call__(self, state: Dict, **kwargs) -> any:
        """Compute action given current state."""
        pass


class PredatorStrategy(Strategy):
    """Base class for predator drift strategies."""

    @abstractmethod
    def __call__(self, state: Dict, **kwargs) -> float:
        """
        Compute predator drift w_t.

        Args:
            state: Environment state dict with keys:
                - 'mid_price': Current price S_t
                - 'inventory': MM's inventory q_t
                - 'time_remaining': Fraction of episode left
                - etc.

        Returns:
            w_t: Drift control
        """
        pass


class MarketMakerStrategy(Strategy):
    """Base class for market maker strategies."""

    @abstractmethod
    def __call__(self, state: Dict, **kwargs) -> Tuple[float, float]:
        """
        Compute MM's bid/ask spreads.

        Args:
            state: Environment state dict

        Returns:
            (delta_bid, delta_ask): Half-spreads as fractions
        """
        pass


class MacroStrategy(Strategy):
    """Base class for macro regime control strategies."""

    @abstractmethod
    def __call__(self, state: Dict, **kwargs) -> Tuple[float, float]:
        """
        Compute macro controls (f, g).

        Args:
            state: Environment state dict

        Returns:
            (f, g): Attack and defense controls in [0, 1]
        """
        pass


# ============================================================================
# Predator Strategies
# ============================================================================

class OptimalPredator(PredatorStrategy):
    """
    Optimal closed-form predator strategy from paper.

    Formula (line 955-957):
        w*(t, q) = -ξγq

    Mean-averting behavior:
    - MM long (q > 0) → drift down (w < 0) to devalue position
    - MM short (q < 0) → drift up (w > 0) to make covering expensive
    """

    def __init__(self, xi: float, gamma: float):
        """
        Initialize optimal predator.

        Args:
            xi: Predator cost coefficient (higher = more expensive to manipulate)
            gamma: MM's risk aversion coefficient
        """
        self.xi = xi
        self.gamma = gamma

    def __call__(self, state: Dict, **kwargs) -> float:
        """
        Compute optimal predator drift.

        Args:
            state: Must contain 'inventory' key

        Returns:
            w_t = -ξγq
        """
        q = float(state['inventory'])
        return -self.xi * self.gamma * q

    def __repr__(self) -> str:
        return f"OptimalPredator(xi={self.xi}, gamma={self.gamma})"


class NoPredator(PredatorStrategy):
    """Baseline: no predator (w_t = 0)."""

    def __call__(self, state: Dict, **kwargs) -> float:
        return 0.0

    def __repr__(self) -> str:
        return "NoPredator()"


# ============================================================================
# Market Maker Strategies
# ============================================================================

class AdversarialAvellanedaStoikov(MarketMakerStrategy):
    """
    Optimal MM strategy with predatory risk adjustment.

    From paper (lines 1030-1034):
        C_i(τ) = γw_i(τ) + γ²ξτ  [Effective risk factor]
        u*_i = (1/γ)ln(1+γ/κ) + (1/2)C_i(τ)  [Optimal spread]
        r_i = S_t - qC_i(τ)  [Reservation price]

    For short horizons (line 1013-1015):
        w_i(τ) ≈ σ_i²τ  [Expected variance, ignoring regime mixing]

    So:
        C_i(τ) ≈ γσ_i²τ + γ²ξτ = γτ(σ_i² + γξ)

    Key insight (line 979): Predator is isomorphic to increased volatility
        σ_eff² = σ_i² + γξ
    """

    def __init__(
        self,
        gamma: float,
        kappa: float,
        xi: float,
        sigma_stable: float = 0.40,
        sigma_volatile: float = 0.80,
        T_horizon: float = 12.0 / 252,  # 12 hours in annual fraction
    ):
        """
        Initialize adversarial AS strategy.

        Args:
            gamma: Risk aversion coefficient
            kappa: Spread sensitivity (order intensity decay)
            xi: Predator cost coefficient
            sigma_stable: Volatility in regime 0 (annualized)
            sigma_volatile: Volatility in regime 1 (annualized)
            T_horizon: Total episode horizon (annual fraction)
        """
        self.gamma = gamma
        self.kappa = kappa
        self.xi = xi
        self.sigma = {0: sigma_stable, 1: sigma_volatile}
        self.T_horizon = T_horizon

        # Monopoly rent (doesn't depend on state)
        self.monopoly_rent = (1.0 / gamma) * np.log(1 + gamma / kappa)

    def __call__(
        self,
        state: Dict,
        regime: Optional[int] = None,
        **kwargs
    ) -> Tuple[float, float]:
        """
        Compute optimal bid/ask spreads.

        Args:
            state: Must contain 'inventory' and 'time_remaining'
            regime: Current regime (0=stable, 1=volatile). If None, use conservative estimate.

        Returns:
            (delta_bid, delta_ask): Half-spreads as fractions
        """
        q = float(state['inventory'])
        time_remaining = float(state['time_remaining'])
        tau = time_remaining * self.T_horizon  # Convert to annual fraction

        # Determine regime volatility
        if regime is not None:
            sigma = self.sigma[regime]
        else:
            # Conservative: use higher volatility if regime unknown
            sigma = max(self.sigma.values())

        # Effective volatility with predatory risk (line 979)
        sigma_eff_sq = sigma**2 + self.gamma * self.xi

        # Effective risk factor C_i(τ) (line 1025)
        C = self.gamma * sigma_eff_sq * tau

        # Optimal symmetric spread (line 1032)
        u_star = self.monopoly_rent + 0.5 * C

        # Reservation price (line 1033)
        # r = S - qC, but we return spreads relative to S
        # So we need to adjust for inventory skew
        inventory_skew = q * C

        # Bid/ask spreads (symmetric around reservation price)
        # bid = r - u* = (S - qC) - u*
        # ask = r + u* = (S - qC) + u*
        #
        # As fraction of S:
        # delta_bid = (S - bid)/S = (S - (S - qC - u*))/S = (qC + u*)/S
        # delta_ask = (ask - S)/S = ((S - qC + u*) - S)/S = (-qC + u*)/S

        # Since we're working in fractions, and S cancels out in the optimal policy,
        # we use the absolute spread formulas then divide by S
        # But actually the paper gives spreads in price units, and we need fractions
        # Let's use the direct formula:

        # For small C relative to S, spreads as fractions:
        delta_bid = u_star + inventory_skew  # Wider when long (q > 0)
        delta_ask = u_star - inventory_skew  # Wider when short (q < 0)

        # Ensure non-negative spreads
        delta_bid = max(delta_bid, 1e-6)
        delta_ask = max(delta_ask, 1e-6)

        return delta_bid, delta_ask

    def __repr__(self) -> str:
        return (f"AdversarialAvellanedaStoikov(gamma={self.gamma}, kappa={self.kappa}, "
                f"xi={self.xi}, sigma={self.sigma})")


class SymmetricSpread(MarketMakerStrategy):
    """Baseline: constant symmetric spread (no inventory adjustment)."""

    def __init__(self, spread: float = 0.001):
        """
        Args:
            spread: Half-spread as fraction (default 10 bps)
        """
        self.spread = spread

    def __call__(self, state: Dict, **kwargs) -> Tuple[float, float]:
        return self.spread, self.spread

    def __repr__(self) -> str:
        return f"SymmetricSpread(spread={self.spread})"


# ============================================================================
# Macro Strategies
# ============================================================================

class ProportionalMacro(MacroStrategy):
    """
    Proportional feedback on regime preferences.

    Simplified from lines 1076-1079. Since we don't have full outer value functions U_i,
    we use heuristics:
    - Attacker (f): Increases when market is calm (wants volatility)
    - Defender (g): Increases when market is volatile (wants stability)
    """

    def __init__(
        self,
        rho_f: float = 1.0,
        rho_g: float = 1.0,
        baseline_f: float = 0.3,
        baseline_g: float = 0.7,
    ):
        """
        Initialize proportional macro strategy.

        Args:
            rho_f: Attack effort cost (higher = less aggressive)
            rho_g: Defense effort cost (higher = less defensive)
            baseline_f: Base attack level
            baseline_g: Base defense level
        """
        self.rho_f = rho_f
        self.rho_g = rho_g
        self.baseline_f = baseline_f
        self.baseline_g = baseline_g

    def __call__(
        self,
        state: Dict,
        regime: Optional[int] = None,
        **kwargs
    ) -> Tuple[float, float]:
        """
        Compute macro controls based on current regime.

        Args:
            state: Environment state
            regime: Current regime (0=stable, 1=volatile)

        Returns:
            (f, g): Attack and defense controls
        """
        if regime is None:
            # No information, use baseline
            return self.baseline_f, self.baseline_g

        # Heuristic: attacker wants to destabilize, defender wants to stabilize
        if regime == 0:  # Currently stable
            # Attacker: increase effort to create volatility
            f = min(self.baseline_f * 1.5, 1.0)
            # Defender: low effort (already stable)
            g = max(self.baseline_g * 0.5, 0.0)
        else:  # Currently volatile
            # Attacker: low effort (already volatile)
            f = max(self.baseline_f * 0.5, 0.0)
            # Defender: increase effort to stabilize
            g = min(self.baseline_g * 1.5, 1.0)

        return f, g

    def __repr__(self) -> str:
        return (f"ProportionalMacro(rho_f={self.rho_f}, rho_g={self.rho_g}, "
                f"baseline_f={self.baseline_f}, baseline_g={self.baseline_g})")


class TimeVaryingMacro(MacroStrategy):
    """
    Time-varying macro strategy (increasing attack over time).

    Simple linear ramp from (f0, g0) to (f1, g1).
    """

    def __init__(
        self,
        f_start: float = 0.3,
        f_end: float = 0.7,
        g_start: float = 0.7,
        g_end: float = 0.3,
    ):
        """
        Args:
            f_start: Initial attack level
            f_end: Final attack level
            g_start: Initial defense level
            g_end: Final defense level
        """
        self.f_start = f_start
        self.f_end = f_end
        self.g_start = g_start
        self.g_end = g_end

    def __call__(self, state: Dict, **kwargs) -> Tuple[float, float]:
        """
        Linearly interpolate controls based on time remaining.

        Args:
            state: Must contain 'time_remaining'
        """
        time_elapsed = 1.0 - float(state['time_remaining'])

        f = self.f_start + time_elapsed * (self.f_end - self.f_start)
        g = self.g_start + time_elapsed * (self.g_end - self.g_start)

        return np.clip(f, 0, 1), np.clip(g, 0, 1)

    def __repr__(self) -> str:
        return (f"TimeVaryingMacro(f: {self.f_start}->{self.f_end}, "
                f"g: {self.g_start}->{self.g_end})")


class ConstantMacro(MacroStrategy):
    """Baseline: constant macro controls."""

    def __init__(self, f: float = 0.5, g: float = 0.5):
        self.f = f
        self.g = g

    def __call__(self, state: Dict, **kwargs) -> Tuple[float, float]:
        return self.f, self.g

    def __repr__(self) -> str:
        return f"ConstantMacro(f={self.f}, g={self.g})"


# ============================================================================
# Strategy Collections
# ============================================================================

class StrategyProfile:
    """
    Complete strategy profile for the game.

    Bundles MM, Predator, and Macro strategies together.
    """

    def __init__(
        self,
        mm_strategy: MarketMakerStrategy,
        predator_strategy: PredatorStrategy,
        macro_strategy: MacroStrategy,
        name: str = "Strategy Profile",
    ):
        """
        Args:
            mm_strategy: Market maker strategy
            predator_strategy: Predator strategy
            macro_strategy: Macro control strategy
            name: Profile name for identification
        """
        self.mm = mm_strategy
        self.predator = predator_strategy
        self.macro = macro_strategy
        self.name = name

    def __repr__(self) -> str:
        return (f"StrategyProfile('{self.name}'):\n"
                f"  MM: {self.mm}\n"
                f"  Predator: {self.predator}\n"
                f"  Macro: {self.macro}")


# ============================================================================
# Pre-defined Profiles
# ============================================================================

def get_optimal_profile(
    gamma: float = 0.01,
    kappa: float = 1.5,
    xi: float = 0.01,
) -> StrategyProfile:
    """
    Optimal closed-form strategies from paper.

    - Predator: w* = -ξγq
    - MM: Adversarial AS with predatory risk
    - Macro: Time-varying (increasing attack)
    """
    return StrategyProfile(
        mm_strategy=AdversarialAvellanedaStoikov(gamma=gamma, kappa=kappa, xi=xi),
        predator_strategy=OptimalPredator(xi=xi, gamma=gamma),
        macro_strategy=TimeVaryingMacro(f_start=0.3, f_end=0.7, g_start=0.7, g_end=0.3),
        name="Optimal (Paper Formulas)",
    )


def get_baseline_profile() -> StrategyProfile:
    """
    Baseline: no predator, constant spread, constant macro.
    """
    return StrategyProfile(
        mm_strategy=SymmetricSpread(spread=0.001),
        predator_strategy=NoPredator(),
        macro_strategy=ConstantMacro(f=0.5, g=0.5),
        name="Baseline (No Game)",
    )


def get_mm_only_profile(gamma: float = 0.01, kappa: float = 1.5) -> StrategyProfile:
    """
    MM plays optimally, no predator, constant macro.
    """
    return StrategyProfile(
        mm_strategy=AdversarialAvellanedaStoikov(gamma=gamma, kappa=kappa, xi=0.0),
        predator_strategy=NoPredator(),
        macro_strategy=ConstantMacro(f=0.5, g=0.5),
        name="MM Only (No Adversary)",
    )
