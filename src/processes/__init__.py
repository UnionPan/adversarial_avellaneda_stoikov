"""
Stochastic Process Implementations

Provides various stochastic processes for financial modeling:
- Geometric Brownian Motion (GBM)
- Heston stochastic volatility
- Jump diffusions (Merton, Kou)
- Levy processes (NIG, VG)
- Regime switching models
- And more

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from .base import (
    StochasticProcess,
    MultiFactorProcess,
    SimulationConfig,
)

from .gbm import GBM
from .heston import Heston
from .cev import CEV
from .ornstein_uhlenbeck import OrnsteinUhlenbeck
from .sabr import SABR

# Jump diffusions
from .merton import MertonJD
from .kou import KouJD

# Levy processes
from .nig import NIG
from .variance_gamma import VarianceGamma

# Multi-asset
from .multi_asset_gbm import MultiAssetGBM

# Regime switching
from .regime_switching import RegimeSwitchingProcess
from .regime_switching_gbm import RegimeSwitchingGBM
from .regime_switching_merton import RegimeSwitchingMerton

__all__ = [
    # Base classes
    'StochasticProcess',
    'MultiFactorProcess',
    'SimulationConfig',

    # Single-asset processes
    'GBM',
    'Heston',
    'CEV',
    'OrnsteinUhlenbeck',
    'SABR',

    # Jump diffusions
    'MertonJD',
    'KouJD',

    # Levy processes
    'NIG',
    'VarianceGamma',

    # Multi-asset
    'MultiAssetGBM',

    # Regime switching
    'RegimeSwitchingProcess',
    'RegimeSwitchingGBM',
    'RegimeSwitchingMerton',
]
