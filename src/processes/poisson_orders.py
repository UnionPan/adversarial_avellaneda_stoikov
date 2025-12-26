"""
Poisson Order Arrival Process for Market Making

Simulates buy/sell order arrivals with intensity dependent on spread.
Follows Avellaneda-Stoikov framework: λ(δ) = λ₀ * exp(-κ * δ)

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import numpy as np
from typing import Tuple


class PoissonOrderGenerator:
    """
    Generate order arrivals via Poisson process with spread-dependent intensity.

    Order arrival intensity: λ(δ) = λ₀ * exp(-κ * δ)
    where:
        δ: half-spread (distance from mid-price)
        λ₀: base arrival intensity (orders per unit time)
        κ: spread sensitivity (higher κ = stronger penalty for wide spreads)

    Usage:
        generator = PoissonOrderGenerator(lambda_0=10.0, kappa=1.5)
        n_buy, n_sell = generator.sample_arrivals(
            delta_bid=0.01,   # 1% below mid
            delta_ask=0.01,   # 1% above mid
            dt=0.001          # 1 second at dt=1/390/252
        )

    Example (Avellaneda-Stoikov):
        # Symmetric spread: δ+ = δ- = δ
        # Optimal spread: δ* ≈ γ*σ²*(T-t) + (1/γ)*log(1 + γ/κ)
        # where γ is risk aversion, σ is volatility, T-t is time horizon
    """

    def __init__(
        self,
        lambda_0: float = 10.0,  # Base intensity (orders/time unit)
        kappa: float = 1.5,      # Spread sensitivity
    ):
        """
        Initialize order generator.

        Args:
            lambda_0: Base arrival intensity (orders per unit time at δ=0)
            kappa: Spread sensitivity parameter (κ > 0)
                   Higher κ means arrivals drop off faster with spread

        Notes:
            - At δ=0 (no spread): intensity = lambda_0
            - At δ=1/κ: intensity = lambda_0 * exp(-1) ≈ 0.37 * lambda_0
            - Typical values: lambda_0 in [1, 100], kappa in [0.5, 5.0]
        """
        if lambda_0 <= 0:
            raise ValueError("lambda_0 must be positive")
        if kappa <= 0:
            raise ValueError("kappa must be positive")

        self.lambda_0 = lambda_0
        self.kappa = kappa

    def intensity(self, delta: float) -> float:
        """
        Compute arrival intensity for given half-spread.

        Args:
            delta: Half-spread (distance from mid-price)

        Returns:
            Arrival intensity λ(δ) = λ₀ * exp(-κ * δ)
        """
        return self.lambda_0 * np.exp(-self.kappa * delta)

    def sample_arrivals(
        self,
        delta_bid: float,
        delta_ask: float,
        dt: float,
    ) -> Tuple[int, int]:
        """
        Sample order arrivals over timestep dt.

        Market maker posts:
            - Bid: mid - delta_bid
            - Ask: mid + delta_ask

        Order arrivals:
            - Buy orders (market orders that hit our ask): Poisson(λ_ask * dt)
            - Sell orders (market orders that hit our bid): Poisson(λ_bid * dt)

        Args:
            delta_bid: Half-spread for bid (≥ 0)
            delta_ask: Half-spread for ask (≥ 0)
            dt: Time step

        Returns:
            n_buy: Number of buy orders (market maker sells at ask)
            n_sell: Number of sell orders (market maker buys at bid)

        Notes:
            - n_buy > 0 → inventory decreases (we sell)
            - n_sell > 0 → inventory increases (we buy)
            - Wider spreads → lower arrival rates → less inventory risk
        """
        if delta_bid < 0 or delta_ask < 0:
            raise ValueError("Half-spreads must be non-negative")
        if dt <= 0:
            raise ValueError("Time step dt must be positive")

        # Compute intensities
        lambda_bid = self.intensity(delta_bid)
        lambda_ask = self.intensity(delta_ask)

        # Sample arrivals from Poisson
        # Expected arrivals = intensity * dt
        n_buy = np.random.poisson(lambda_ask * dt)
        n_sell = np.random.poisson(lambda_bid * dt)

        return int(n_buy), int(n_sell)

    def expected_arrivals(
        self,
        delta_bid: float,
        delta_ask: float,
        dt: float,
    ) -> Tuple[float, float]:
        """
        Compute expected number of arrivals (for analysis/testing).

        Args:
            delta_bid: Half-spread for bid
            delta_ask: Half-spread for ask
            dt: Time step

        Returns:
            expected_buy: E[n_buy] = λ_ask * dt
            expected_sell: E[n_sell] = λ_bid * dt
        """
        lambda_bid = self.intensity(delta_bid)
        lambda_ask = self.intensity(delta_ask)

        return lambda_ask * dt, lambda_bid * dt

    def __repr__(self) -> str:
        return f"PoissonOrderGenerator(lambda_0={self.lambda_0}, kappa={self.kappa})"
