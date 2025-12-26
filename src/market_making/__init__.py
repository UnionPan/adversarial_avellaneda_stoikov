"""
Market Making Game Module

Double-layer Avellaneda-Stoikov market making game with:
- Inner layer: Market maker vs predator (15s timestep)
- Outer layer: Macro players control regime switching (30min frequency)
- Closed-form strategies from paper (equations 955, 1030-1034, 1076-1079)
- Multi-path PnL simulator with histograms

Features:
- 12-hour trading horizon
- Predator drift control: w* = -ξγq
- Modified AS with predatory risk: σ_eff² = σ² + γξ
- Time-varying macro controls
- Monte Carlo PnL comparison

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from .market_making_env import MarketMakingEnv, make_market_making_env
from .strategies import (
    # Base classes
    Strategy,
    PredatorStrategy,
    MarketMakerStrategy,
    MacroStrategy,
    # Predator strategies
    OptimalPredator,
    NoPredator,
    # MM strategies
    AdversarialAvellanedaStoikov,
    SymmetricSpread,
    # Macro strategies
    TimeVaryingMacro,
    ProportionalMacro,
    ConstantMacro,
    # Profiles
    StrategyProfile,
    get_optimal_profile,
    get_baseline_profile,
    get_mm_only_profile,
)
from .pnl_simulator import PnLSimulator, SimulationResult, quick_comparison

__all__ = [
    # Environment
    'MarketMakingEnv',
    'make_market_making_env',
    # Base strategy classes
    'Strategy',
    'PredatorStrategy',
    'MarketMakerStrategy',
    'MacroStrategy',
    # Predator strategies
    'OptimalPredator',
    'NoPredator',
    # MM strategies
    'AdversarialAvellanedaStoikov',
    'SymmetricSpread',
    # Macro strategies
    'TimeVaryingMacro',
    'ProportionalMacro',
    'ConstantMacro',
    # Profiles
    'StrategyProfile',
    'get_optimal_profile',
    'get_baseline_profile',
    'get_mm_only_profile',
    # Simulator
    'PnLSimulator',
    'SimulationResult',
    'quick_comparison',
]
