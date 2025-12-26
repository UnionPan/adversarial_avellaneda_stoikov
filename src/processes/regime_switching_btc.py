"""
Regime-Switching BTC Process with Controlled Transition Rates

Two-regime model for Bitcoin price dynamics:
- Regime 0: Stabilizing (low volatility, positive drift)
- Regime 1: Destabilizing (high volatility, volatile drift)

Macro players control regime transition rates via affine control:
    μij(f, g) = μ0_ij + f · λatt_ij - g · λstab_ij

where:
    - μ0: base transition rate matrix (calibrated from data)
    - λatt: attack influence matrix (promotes destabilizing)
    - λstab: stabilize influence matrix (promotes stabilizing)
    - f ∈ [0,1]: macro player 1 control (attacker)
    - g ∈ [0,1]: macro player 2 control (defender)

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import numpy as np
from typing import Dict, Tuple, Optional
from .regime_switching_gbm import RegimeSwitchingGBM


class RegimeSwitchingBTC(RegimeSwitchingGBM):
    """
    Regime-switching BTC with controlled transition rates.

    dS_t = μ(regime_t) * S_t dt + σ(regime_t) * S_t dW_t

    where regime_t switches according to controlled transition matrix:
        Q(f, g) = Q₀ + f * Λ_att - g * Λ_stab

    Usage:
        # Setup base transition matrix and control matrices
        mu0 = np.array([[-0.5, 0.5],    # From stable to volatile: 0.5/day
                        [1.0, -1.0]])    # From volatile to stable: 1.0/day

        lambda_att = np.array([[0.0, 0.3],   # Increase stable → volatile
                               [0.0, 0.0]])   # (attack only affects exit from stable)

        lambda_stab = np.array([[0.0, 0.0],
                                [0.5, 0.0]])  # Increase volatile → stable

        # Create process
        btc = RegimeSwitchingBTC(
            mu0=mu0,
            lambda_att=lambda_att,
            lambda_stab=lambda_stab,
            regime_params={
                'mu': [0.10, -0.05],      # Regime drifts (annualized)
                'sigma': [0.40, 0.80],    # Regime vols (annualized)
            },
        )

        # Macro players set controls
        btc.set_controls(f=0.7, g=0.3)  # Attacker strong, defender weak

        # Simulate
        t_grid, paths = btc.simulate(
            X0=100.0,
            T=30/252,  # 30 days
            dt=1/252,  # 1 day steps
            n_paths=1000
        )
    """

    def __init__(
        self,
        mu0: np.ndarray,                   # Base 2x2 transition matrix
        lambda_att: np.ndarray,            # Attack influence matrix
        lambda_stab: np.ndarray,           # Stabilize influence matrix
        regime_params: Dict[str, list],    # {'mu': [...], 'sigma': [...]}
        name: str = "RegimeSwitchingBTC",
    ):
        """
        Initialize controlled regime-switching BTC process.

        Args:
            mu0: Base continuous-time transition rate matrix (2x2)
                 Entry μ0[i,j] is rate of transition from regime i to j (i ≠ j)
                 Diagonal μ0[i,i] = -sum(μ0[i, j!=i]) (exit rate from regime i)

            lambda_att: Attack influence matrix (2x2)
                        Increases transitions toward destabilizing regime
                        Typically non-zero only for stable → volatile transition

            lambda_stab: Stabilize influence matrix (2x2)
                         Increases transitions toward stabilizing regime
                         Typically non-zero only for volatile → stable transition

            regime_params: Dictionary with regime-dependent parameters
                           {'mu': [mu_0, mu_1], 'sigma': [sigma_0, sigma_1]}
                           - Regime 0: Stabilizing (low vol)
                           - Regime 1: Destabilizing (high vol)

            name: Process name

        Notes:
            - All transition matrices should have row sums = 0
            - Rates are instantaneous (not probabilities)
            - Typical values:
              - μ0 diagonal: -1 to -0.1 (mean holding time = 1/|μ| days)
              - λatt, λstab: 0.1 to 1.0 (fraction of base rate)
        """
        # Initialize parent with 2 regimes
        super().__init__(n_regimes=2, name=name)

        # Validate matrices
        self._validate_transition_matrix(mu0, "mu0", require_negative_diagonal=True)
        self._validate_transition_matrix(lambda_att, "lambda_att", require_negative_diagonal=False)
        self._validate_transition_matrix(lambda_stab, "lambda_stab", require_negative_diagonal=False)

        # Store control matrices
        self.mu0 = mu0.copy()
        self.lambda_att = lambda_att.copy()
        self.lambda_stab = lambda_stab.copy()

        # Set regime parameters (drift and vol for each regime)
        if 'mu' not in regime_params or 'sigma' not in regime_params:
            raise ValueError("regime_params must contain 'mu' and 'sigma' keys")

        self.set_regime_params('mu', regime_params['mu'])
        self.set_regime_params('sigma', regime_params['sigma'])

        # Initialize with base transition matrix (f=g=0)
        self.set_transition_matrix(self.mu0)

        # Current controls
        self.f = 0.0
        self.g = 0.0

    def _validate_transition_matrix(
        self,
        Q: np.ndarray,
        name: str,
        require_negative_diagonal: bool = True
    ):
        """Validate transition matrix structure."""
        if Q.shape != (2, 2):
            raise ValueError(f"{name} must be 2x2 matrix, got shape {Q.shape}")

        # Check row sums approximately zero (within numerical tolerance)
        row_sums = Q.sum(axis=1)
        if not np.allclose(row_sums, 0.0, atol=1e-6):
            raise ValueError(
                f"{name} rows must sum to zero (generator matrix property). "
                f"Got row sums: {row_sums}"
            )

        # Check diagonal is negative (only for base transition matrix)
        if require_negative_diagonal:
            if Q[0, 0] > 0 or Q[1, 1] > 0:
                raise ValueError(f"{name} diagonal entries must be negative (exit rates)")

    def set_controls(self, f: float, g: float):
        """
        Update transition matrix based on macro player controls.

        μij(f, g) = μ0_ij + f · λatt_ij - g · λstab_ij

        Args:
            f: Macro player 1 control (attacker) ∈ [0, 1]
               Higher f → more transitions to destabilizing regime
            g: Macro player 2 control (defender) ∈ [0, 1]
               Higher g → more transitions to stabilizing regime

        Notes:
            - f and g can work against each other
            - Resulting matrix must remain valid (non-negative off-diagonal)
        """
        if not (0 <= f <= 1):
            raise ValueError(f"f must be in [0, 1], got {f}")
        if not (0 <= g <= 1):
            raise ValueError(f"g must be in [0, 1], got {g}")

        # Update controls
        self.f = f
        self.g = g

        # Compute controlled transition matrix
        Q_controlled = self.mu0 + f * self.lambda_att - g * self.lambda_stab

        # Ensure off-diagonal entries remain non-negative
        # (Transition rates cannot be negative)
        for i in range(2):
            for j in range(2):
                if i != j and Q_controlled[i, j] < 0:
                    # Clip to zero and adjust diagonal
                    Q_controlled[i, i] += Q_controlled[i, j]  # Add back to diagonal
                    Q_controlled[i, j] = 0.0

        # Update transition matrix
        self.set_transition_matrix(Q_controlled)

    def get_current_controls(self) -> Tuple[float, float]:
        """Return current macro controls (f, g)."""
        return self.f, self.g

    def get_regime_info(self, regime: int) -> Dict[str, float]:
        """
        Get drift and volatility for specific regime.

        Args:
            regime: Regime index (0 = stabilizing, 1 = destabilizing)

        Returns:
            Dictionary with 'mu' and 'sigma' for the regime
        """
        if regime not in [0, 1]:
            raise ValueError(f"regime must be 0 or 1, got {regime}")

        return {
            'mu': self.regime_params['mu'][regime],
            'sigma': self.regime_params['sigma'][regime],
        }

    def expected_holding_time(self, regime: int) -> float:
        """
        Compute expected holding time in given regime.

        E[τ] = 1 / |μ[regime, regime]|

        Args:
            regime: Regime index

        Returns:
            Expected holding time (in same units as transition matrix)
        """
        exit_rate = -self.transition_matrix[regime, regime]
        if exit_rate < 1e-12:
            return np.inf
        return 1.0 / exit_rate

    def __repr__(self) -> str:
        return (
            f"RegimeSwitchingBTC(\n"
            f"  regimes: {self.n_regimes},\n"
            f"  controls: f={self.f:.2f}, g={self.g:.2f},\n"
            f"  regime 0 (stable): μ={self.regime_params['mu'][0]:.3f}, "
            f"σ={self.regime_params['sigma'][0]:.3f},\n"
            f"  regime 1 (volatile): μ={self.regime_params['mu'][1]:.3f}, "
            f"σ={self.regime_params['sigma'][1]:.3f},\n"
            f"  Q=\n{self.transition_matrix}\n"
            f")"
        )


def create_default_btc_process(
    vol_low: float = 0.40,    # 40% annualized vol (stable regime)
    vol_high: float = 0.80,   # 80% annualized vol (volatile regime)
    mu_low: float = 0.10,     # 10% drift (stable regime)
    mu_high: float = -0.05,   # -5% drift (volatile regime)
    base_switch_rate: float = 0.5,  # Switch ~once every 2 days on average
    control_magnitude: float = 0.3,  # Control influence (30% of base rate)
) -> RegimeSwitchingBTC:
    """
    Create default BTC regime-switching process with reasonable parameters.

    Args:
        vol_low: Volatility in stabilizing regime (annualized)
        vol_high: Volatility in destabilizing regime (annualized)
        mu_low: Drift in stabilizing regime (annualized)
        mu_high: Drift in destabilizing regime (annualized)
        base_switch_rate: Base transition rate (switches per day)
        control_magnitude: Magnitude of macro control influence

    Returns:
        Configured RegimeSwitchingBTC process
    """
    # Base transition matrix (symmetric switching)
    mu0 = np.array([
        [-base_switch_rate, base_switch_rate],
        [base_switch_rate, -base_switch_rate],
    ])

    # Attack matrix: promotes stable → volatile transition
    lambda_att = np.array([
        [-control_magnitude, control_magnitude],
        [0.0, 0.0],
    ])

    # Stabilize matrix: promotes volatile → stable transition
    lambda_stab = np.array([
        [0.0, 0.0],
        [-control_magnitude, control_magnitude],
    ])

    return RegimeSwitchingBTC(
        mu0=mu0,
        lambda_att=lambda_att,
        lambda_stab=lambda_stab,
        regime_params={
            'mu': [mu_low, mu_high],
            'sigma': [vol_low, vol_high],
        },
    )
