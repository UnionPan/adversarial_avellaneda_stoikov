"""
Simulation Environments for Reinforcement Learning

Provides gym-compatible environments for:
- Heston stochastic volatility model with multi-dimensional trading
- Option hedging and trading on fixed grid
- Portfolio management under partial observability (POMDP)

Features:
- Professional processes.Heston simulation (Milstein/Euler)
- Multi-dimensional action space (underlying + options)
- Fixed grid representation (moneyness × TTM)
- Synthetic equity option chains with realistic pricing
- T=246 steps, dt=1/252 (~1 trading year)

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from .heston_env import HestonEnv, HestonParams, Liability, make_heston_env

__all__ = [
    'HestonEnv',
    'HestonParams',
    'Liability',
    'make_heston_env',
]
