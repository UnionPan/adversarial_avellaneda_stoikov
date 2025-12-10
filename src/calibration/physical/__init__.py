"""
Physical (P-measure) calibrators.

Calibrate models from historical price data (OHLCV).
Used for real-world forecasting, risk management, and counterfactual simulation.

Key characteristic: Drift μ is estimated from historical returns (not set to r).
"""

from .gbm_calibrator import GBMCalibrator, GBMCalibrationResult
from .ou_calibrator import OUCalibrator, OUCalibrationResult
from .volatility import (
    VolatilityEstimate,
    VolatilityEstimator,
    GARCHEstimator,
    compare_volatility_methods,
)
from .regime_switching_calibrator import (
    RegimeSwitchingCalibrator,
    RegimeSwitchingSimulator,
    RegimeSwitchingCalibrationResult,
    RegimeParameters,
)

__all__ = [
    # GBM
    'GBMCalibrator',
    'GBMCalibrationResult',

    # Ornstein-Uhlenbeck
    'OUCalibrator',
    'OUCalibrationResult',

    # Volatility estimators
    'VolatilityEstimate',
    'VolatilityEstimator',
    'GARCHEstimator',
    'compare_volatility_methods',

    # Regime-switching
    'RegimeSwitchingCalibrator',
    'RegimeSwitchingSimulator',
    'RegimeSwitchingCalibrationResult',
    'RegimeParameters',
]
