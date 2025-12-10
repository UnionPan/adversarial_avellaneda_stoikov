"""
Pricer module - Comprehensive derivative pricing methods

author: Yunian Pan
email: yp1170@nyu.edu
"""

# Base classes
from .base import Pricer, PricingResult

# Monte Carlo pricer
from .monte_carlo import MonteCarloPricer

# Analytical pricers
from .analytical import BlackScholesPricer, HestonAnalyticalPricer

# Fourier-based pricers
from .fourier import COSPricer, CarrMadanPricer

# PDE pricers
from .finite_difference import FiniteDifferencePricer, AdaptiveFiniteDifferencePricer
from .finite_element import FiniteElementPricer, HighOrderFiniteElementPricer

__all__ = [
    # Base
    "Pricer",
    "PricingResult",
    # Monte Carlo
    "MonteCarloPricer",
    # Analytical
    "BlackScholesPricer",
    "HestonAnalyticalPricer",
    # Fourier
    "COSPricer",
    "CarrMadanPricer",
    # PDE
    "FiniteDifferencePricer",
    "AdaptiveFiniteDifferencePricer",
    "FiniteElementPricer",
    "HighOrderFiniteElementPricer",
]
