"""
Volatility Estimation from Historical Data

Implements industry-standard volatility estimators:
- Simple historical volatility (rolling window)
- EWMA (RiskMetrics)
- GARCH(1,1) (Engle-Bollerslev)

These are used for:
1. Risk management (VaR, volatility targeting)
2. Parameter initialization for calibration
3. Validation of implied volatility

author: Yunian Pan
email: yp1170@nyu.edu
"""

import numpy as np
from scipy import optimize
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import warnings


@dataclass
class VolatilityEstimate:
    """Container for volatility estimation results."""

    # Volatility time series (annualized)
    volatility: np.ndarray

    # Metadata
    method: str
    annualization_factor: float

    # Parameters (method-dependent)
    params: Optional[Dict] = None

    @property
    def current_vol(self) -> float:
        """Most recent volatility estimate."""
        return self.volatility[-1]

    @property
    def average_vol(self) -> float:
        """Average volatility over the period."""
        return np.mean(self.volatility)

    def __repr__(self) -> str:
        return (
            f"VolatilityEstimate(method={self.method}, "
            f"current={self.current_vol:.2%}, "
            f"average={self.average_vol:.2%})"
        )


class VolatilityEstimator:
    """
    Historical volatility estimation using various methods.

    Example:
        >>> estimator = VolatilityEstimator(method='ewma', lambda_=0.94)
        >>> returns = np.random.randn(252) * 0.01
        >>> result = estimator.estimate(returns, annualize=True)
        >>> print(f"Current vol: {result.current_vol:.2%}")
    """

    def __init__(
        self,
        method: str = 'simple',
        window: int = 252,
        lambda_: float = 0.94,
        min_periods: int = 20,
    ):
        """
        Initialize volatility estimator.

        Args:
            method: Estimation method ('simple', 'ewma', 'garch')
            window: Rolling window size for simple method (default: 252 days = 1 year)
            lambda_: Decay factor for EWMA (default: 0.94, RiskMetrics standard)
            min_periods: Minimum observations required
        """
        self.method = method.lower()
        self.window = window
        self.lambda_ = lambda_
        self.min_periods = min_periods

        if self.method not in ['simple', 'ewma', 'garch']:
            raise ValueError(f"Unknown method: {method}. Use 'simple', 'ewma', or 'garch'")

    def estimate(
        self,
        returns: np.ndarray,
        annualize: bool = True,
        periods_per_year: int = 252,
    ) -> VolatilityEstimate:
        """
        Estimate volatility from return series.

        Args:
            returns: Return series (NOT prices - use compute_returns() first)
            annualize: Whether to annualize volatility
            periods_per_year: Number of periods per year (252 for daily, 12 for monthly)

        Returns:
            VolatilityEstimate with volatility time series
        """
        returns = np.asarray(returns).flatten()

        if len(returns) < self.min_periods:
            raise ValueError(
                f"Insufficient data: {len(returns)} < {self.min_periods} required"
            )

        # Compute volatility based on method
        if self.method == 'simple':
            vol = self._simple_vol(returns)
        elif self.method == 'ewma':
            vol = self._ewma_vol(returns)
        elif self.method == 'garch':
            raise ValueError("Use GARCHEstimator class for GARCH estimation")
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Annualize if requested
        annualization_factor = np.sqrt(periods_per_year) if annualize else 1.0
        vol = vol * annualization_factor

        params = {
            'window': self.window,
            'lambda': self.lambda_,
        }

        return VolatilityEstimate(
            volatility=vol,
            method=self.method,
            annualization_factor=annualization_factor,
            params=params,
        )

    def _simple_vol(self, returns: np.ndarray) -> np.ndarray:
        """
        Simple rolling standard deviation.

        σ_t = sqrt(1/(n-1) * Σ(r_i - r_mean)²) over window [t-n+1, t]

        Pros: Easy to understand, unbiased estimator
        Cons: Equal weight to all observations, sensitive to window size
        """
        n = len(returns)
        vol = np.full(n, np.nan)

        for i in range(self.window - 1, n):
            window_returns = returns[i - self.window + 1 : i + 1]
            vol[i] = np.std(window_returns, ddof=1)

        return vol

    def _ewma_vol(self, returns: np.ndarray) -> np.ndarray:
        """
        Exponentially Weighted Moving Average (RiskMetrics).

        Recursive formula:
        σ²_t = λ * σ²_{t-1} + (1-λ) * r²_t

        where λ = decay factor (0.94 for daily data)

        Decay half-life: log(0.5) / log(λ) ≈ 11.4 days for λ=0.94

        Pros: Adaptive, no window parameter, recent data weighted more
        Cons: One parameter to tune (but 0.94 is industry standard)
        """
        n = len(returns)
        variance = np.zeros(n)

        # Initialize with simple variance over first window
        init_window = min(self.min_periods, len(returns))
        variance[0] = np.var(returns[:init_window], ddof=1)

        # Recursive update
        for t in range(1, n):
            variance[t] = self.lambda_ * variance[t-1] + (1 - self.lambda_) * returns[t]**2

        return np.sqrt(variance)


class GARCHEstimator:
    """
    GARCH(1,1) Volatility Estimation

    Industry standard for volatility modeling.

    Model:
        r_t = σ_t * ε_t,  ε_t ~ N(0, 1)
        σ²_t = ω + α*r²_{t-1} + β*σ²_{t-1}

    Constraints:
        ω > 0, α ≥ 0, β ≥ 0
        α + β < 1 (stationarity)

    Long-run variance: σ²_LR = ω / (1 - α - β)
    Persistence: α + β (close to 1 → high persistence)

    Example:
        >>> estimator = GARCHEstimator()
        >>> result = estimator.estimate(returns)
        >>> print(f"ω={result.params['omega']:.6f}")
        >>> print(f"α={result.params['alpha']:.4f}")
        >>> print(f"β={result.params['beta']:.4f}")
        >>> print(f"Persistence: {result.persistence:.4f}")
    """

    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ):
        """
        Initialize GARCH estimator.

        Args:
            max_iter: Maximum iterations for MLE optimization
            tol: Convergence tolerance
        """
        self.max_iter = max_iter
        self.tol = tol

    def estimate(
        self,
        returns: np.ndarray,
        annualize: bool = True,
        periods_per_year: int = 252,
    ) -> VolatilityEstimate:
        """
        Estimate GARCH(1,1) model via Maximum Likelihood.

        Args:
            returns: Return series
            annualize: Whether to annualize volatility
            periods_per_year: Number of periods per year

        Returns:
            VolatilityEstimate with fitted volatility and parameters
        """
        returns = np.asarray(returns).flatten()

        if len(returns) < 50:
            raise ValueError("GARCH requires at least 50 observations")

        # Initial parameter guess
        # Common starting values based on unconditional moments
        uncond_var = np.var(returns, ddof=1)

        # Initial guess: ω=0.01*var, α=0.1, β=0.85 (typical values)
        x0 = np.array([
            0.01 * uncond_var,  # omega
            0.10,                # alpha
            0.85,                # beta
        ])

        # Parameter bounds
        bounds = [
            (1e-6, None),  # omega > 0
            (0.0, 1.0),    # 0 ≤ alpha < 1
            (0.0, 1.0),    # 0 ≤ beta < 1
        ]

        # Constraint: alpha + beta < 1 (stationarity)
        constraints = {
            'type': 'ineq',
            'fun': lambda x: 0.9999 - (x[1] + x[2])  # alpha + beta < 1
        }

        # Optimize
        result = optimize.minimize(
            fun=self._negative_log_likelihood,
            x0=x0,
            args=(returns,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iter, 'ftol': self.tol},
        )

        if not result.success:
            warnings.warn(f"GARCH optimization did not converge: {result.message}")

        omega, alpha, beta = result.x

        # Compute fitted conditional volatility
        conditional_var = self._compute_conditional_variance(returns, omega, alpha, beta)
        conditional_vol = np.sqrt(conditional_var)

        # Annualize if requested
        annualization_factor = np.sqrt(periods_per_year) if annualize else 1.0
        conditional_vol = conditional_vol * annualization_factor

        # Compute diagnostics
        long_run_var = omega / (1 - alpha - beta)
        persistence = alpha + beta
        half_life = np.log(0.5) / np.log(persistence) if persistence > 0 else np.inf

        params = {
            'omega': omega,
            'alpha': alpha,
            'beta': beta,
            'long_run_vol': np.sqrt(long_run_var) * annualization_factor,
            'persistence': persistence,
            'half_life': half_life,
            'log_likelihood': -result.fun,
            'converged': result.success,
        }

        return VolatilityEstimate(
            volatility=conditional_vol,
            method='garch',
            annualization_factor=annualization_factor,
            params=params,
        )

    def _negative_log_likelihood(
        self,
        params: np.ndarray,
        returns: np.ndarray,
    ) -> float:
        """
        Negative log-likelihood for GARCH(1,1).

        L = -0.5 * Σ[log(2π) + log(σ²_t) + r²_t/σ²_t]

        Minimize negative LL = maximize LL
        """
        omega, alpha, beta = params

        # Compute conditional variances
        conditional_var = self._compute_conditional_variance(returns, omega, alpha, beta)

        # Log-likelihood (ignoring constant term)
        # LL = -0.5 * Σ[log(σ²_t) + r²_t/σ²_t]
        log_likelihood = -0.5 * np.sum(
            np.log(conditional_var) + returns**2 / conditional_var
        )

        return -log_likelihood  # Return negative for minimization

    def _compute_conditional_variance(
        self,
        returns: np.ndarray,
        omega: float,
        alpha: float,
        beta: float,
    ) -> np.ndarray:
        """
        Compute conditional variance series σ²_t.

        σ²_t = ω + α*r²_{t-1} + β*σ²_{t-1}
        """
        n = len(returns)
        variance = np.zeros(n)

        # Initialize with unconditional variance
        # E[σ²] = ω / (1 - α - β)
        if alpha + beta < 1:
            variance[0] = omega / (1 - alpha - beta)
        else:
            variance[0] = np.var(returns, ddof=1)

        # Recursive update
        for t in range(1, n):
            variance[t] = omega + alpha * returns[t-1]**2 + beta * variance[t-1]

        return variance


def compare_volatility_methods(
    returns: np.ndarray,
    window: int = 252,
    lambda_: float = 0.94,
    annualize: bool = True,
) -> Dict[str, VolatilityEstimate]:
    """
    Compare different volatility estimation methods.

    Useful for understanding method differences and choosing the best one.

    Args:
        returns: Return series
        window: Window for simple method
        lambda_: Decay factor for EWMA
        annualize: Annualize results

    Returns:
        Dictionary with results from each method
    """
    results = {}

    # Simple
    simple_est = VolatilityEstimator(method='simple', window=window)
    results['simple'] = simple_est.estimate(returns, annualize=annualize)

    # EWMA
    ewma_est = VolatilityEstimator(method='ewma', lambda_=lambda_)
    results['ewma'] = ewma_est.estimate(returns, annualize=annualize)

    # GARCH
    try:
        garch_est = GARCHEstimator()
        results['garch'] = garch_est.estimate(returns, annualize=annualize)
    except Exception as e:
        warnings.warn(f"GARCH estimation failed: {e}")
        results['garch'] = None

    return results
