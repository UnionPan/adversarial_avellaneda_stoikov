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
    parkinson_volatility,
    garman_klass_volatility,
    rogers_satchell_volatility,
    yang_zhang_volatility,
    compare_ohlc_estimators,
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
    'parkinson_volatility',
    'garman_klass_volatility',
    'rogers_satchell_volatility',
    'yang_zhang_volatility',
    'compare_ohlc_estimators',

    # Regime-switching
    'RegimeSwitchingCalibrator',
    'RegimeSwitchingSimulator',
    'RegimeSwitchingCalibrationResult',
    'RegimeParameters',
]
