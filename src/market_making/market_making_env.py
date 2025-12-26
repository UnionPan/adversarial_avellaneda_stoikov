"""
Market Making Environment - Multi-Agent Game

Double-layer Avellaneda-Stoikov market making game:
- Inner layer: Market maker vs predator (15s timestep)
- Outer layer: Macro players control regime switching (30min timestep)
- Price: dS_t = w_t dt + σ(I_t) dW_t
- Inventory: Poisson arrivals with spread sensitivity
- Objective: Maximize CARA utility

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass
import warnings

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

import sys
sys.path.insert(0, 'src')

from processes.regime_switching_btc import RegimeSwitchingBTC, create_default_btc_process
from processes.poisson_orders import PoissonOrderGenerator
from processes import SimulationConfig
from utils.utility import CARAUtility


@dataclass
class MarketMakerState:
    """Market maker state variables."""
    S: float           # Mid-price
    q: int             # Inventory (signed)
    cash: float        # Cash position
    t: int             # Current timestep
    regime: int        # Current price regime (0=stable, 1=volatile)
    
    def wealth(self) -> float:
        """Total wealth = cash + inventory * price."""
        return self.cash + self.q * self.S


class MarketMakingEnv(gym.Env):
    """
    Single-agent market making environment.
    
    Agent controls bid/ask spreads to maximize CARA utility while managing:
    - Adverse selection (regime switching affects price)
    - Inventory risk (accumulation)
    - Order arrival randomness (Poisson)
    
    Observation Space:
        Dict with:
        - 'mid_price': Current BTC mid-price (normalized)
        - 'inventory': Current position in BTC
        - 'cash': Cash position
        - 'time_remaining': Fraction of episode remaining [0, 1]
        - 'recent_returns': Last 10 returns (for vol estimation)
        - 'wealth': cash + inventory * mid_price
    
    Action Space:
        Dict with:
        - 'delta_bid': Bid half-spread (≥ 0, as fraction of price)
        - 'delta_ask': Ask half-spread (≥ 0, as fraction of price)
        
        Example: S=100, delta_bid=0.001 → bid at 99.9
                 S=100, delta_ask=0.001 → ask at 100.1
    
    Reward:
        CARA utility: U(W) = -exp(-γ * W)
        where W = cash + inventory * mid_price
        
        Running reward: utility at each step
        Terminal reward: liquidate inventory, final utility
    
    Dynamics:
        1. Post quotes: bid = S - delta_bid*S, ask = S + delta_ask*S
        2. Sample arrivals: Poisson(λ(delta) * dt)
        3. Update inventory: q += n_sell - n_buy
        4. Update cash: cash += n_sell*bid - n_buy*ask
        5. Price step: S evolves via regime-switching BTC
    
    Example:
        env = make_market_making_env(
            S_0=50000.0,           # BTC at $50k
            gamma=0.01,            # Risk aversion
            max_steps=1440,        # 24 hours (1 min steps)
            inventory_limit=10,    # Max ±10 BTC
        )
        
        obs, info = env.reset()
        
        for t in range(1440):
            # Simple strategy: 10 bps spread, symmetric
            action = {'delta_bid': 0.001, 'delta_ask': 0.001}
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        print(f"Final wealth: ${obs['wealth']:.2f}")
        print(f"Final inventory: {obs['inventory']:.4f} BTC")
    """
    
    metadata = {'render_modes': ['human'], 'render_fps': 60}
    
    def __init__(
        self,
        # Price process
        btc_process: Optional[RegimeSwitchingBTC] = None,
        S_0: float = 50000.0,              # Initial BTC price ($)
        
        # Order arrival
        order_generator: Optional[PoissonOrderGenerator] = None,
        lambda_0: float = 100.0 * 252,     # Base order intensity (annualized for dt scaling)
        kappa: float = 1.5,                # Spread sensitivity
        
        # Market maker settings
        gamma: float = 0.01,               # CARA risk aversion
        initial_cash: float = 100000.0,    # Starting cash ($)
        initial_inventory: int = 0,        # Starting inventory (BTC)
        inventory_limit: float = 10.0,     # Max absolute inventory
        
        # Episode settings
        max_steps: int = 2880,             # 12 hours * 4 steps/min (15s each)
        dt_minutes: float = 0.25,          # Timestep (minutes) = 15 seconds
        macro_freq: int = 120,             # Macro decision frequency (every 30 min = 120 steps)

        # Predator settings
        xi: float = 0.01,                  # Predator cost coefficient

        # Initial macro controls
        macro_f: float = 0.5,              # Initial attack control
        macro_g: float = 0.5,              # Initial defense control
        
        # Terminal liquidation
        terminal_liquidation_cost: float = 0.001,  # 10 bps to liquidate
        
        # Rendering
        render_mode: Optional[str] = None,
    ):
        """
        Initialize market making environment.

        Args:
            btc_process: Pre-configured BTC process (if None, uses default)
            S_0: Initial BTC price
            order_generator: Pre-configured order generator
            lambda_0: Base order arrival intensity
            kappa: Spread sensitivity in order arrival
            gamma: CARA risk aversion coefficient
            initial_cash: Starting cash position
            initial_inventory: Starting inventory
            inventory_limit: Maximum absolute inventory
            max_steps: Episode length (timesteps), default 2880 = 12 hours at 15s
            dt_minutes: Time step size (minutes), default 0.25 = 15 seconds
            macro_freq: How often macro controls can change (in steps)
            xi: Predator drift cost coefficient (higher = more expensive)
            macro_f: Initial attack control for regime switching
            macro_g: Initial defense control for regime switching
            terminal_liquidation_cost: Cost to liquidate inventory (as fraction)
            render_mode: 'human' for console output
        """
        super().__init__()
        
        # Price process
        if btc_process is None:
            self.btc_process = create_default_btc_process()
        else:
            self.btc_process = btc_process
        
        # Macro control settings
        self.macro_freq = macro_freq
        self.macro_f = macro_f  # Current macro controls
        self.macro_g = macro_g
        self.btc_process.set_controls(macro_f, macro_g)

        # Predator settings
        self.xi = xi

        self.S_0 = S_0
        
        # Order arrival
        if order_generator is None:
            self.order_generator = PoissonOrderGenerator(
                lambda_0=lambda_0,
                kappa=kappa,
            )
        else:
            self.order_generator = order_generator
        
        # Settings
        self.gamma = gamma
        self.utility = CARAUtility(gamma=gamma)
        self.initial_cash = initial_cash
        self.initial_inventory = initial_inventory
        self.inventory_limit = inventory_limit
        self.max_steps = max_steps
        self.dt_minutes = dt_minutes
        self.dt = dt_minutes / (252 * 1440)  # Convert to annual fraction
        self.terminal_liquidation_cost = terminal_liquidation_cost
        self.render_mode = render_mode
        
        # State (no pre-simulation, evolves dynamically)
        self.state: Optional[MarketMakerState] = None
        self.current_regime: int = 0  # Current regime (tracked separately)
        
        # History
        self.history: Dict[str, List] = {
            'mid_price': [],
            'inventory': [],
            'cash': [],
            'wealth': [],
            'bid': [],
            'ask': [],
            'spread': [],
            'n_buy': [],
            'n_sell': [],
            'regime': [],
            'predator_drift': [],  # w_t at each step
            'macro_f': [],         # f_t at each step
            'macro_g': [],         # g_t at each step
            'reward': [],
        }
        
        # Observation space
        self.observation_space = spaces.Dict({
            'mid_price': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'inventory': spaces.Box(low=-inventory_limit, high=inventory_limit, shape=(1,), dtype=np.float32),
            'cash': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'time_remaining': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'recent_returns': spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
            'wealth': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        })
        
        # Action space: bid/ask half-spreads as fractions of price
        self.action_space = spaces.Dict({
            'delta_bid': spaces.Box(low=0, high=0.1, shape=(1,), dtype=np.float32),   # Up to 10%
            'delta_ask': spaces.Box(low=0, high=0.1, shape=(1,), dtype=np.float32),
        })
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            observation: Initial observation dict
            info: Additional info
        """
        if seed is not None:
            np.random.seed(seed)

        # Initialize regime (start in stable regime 0)
        self.current_regime = 0
        self.btc_process.set_controls(self.macro_f, self.macro_g)

        # Initialize state
        self.state = MarketMakerState(
            S=self.S_0,
            q=self.initial_inventory,
            cash=self.initial_cash,
            t=0,
            regime=self.current_regime,
        )

        # Clear history
        self.history = {k: [] for k in self.history.keys()}
        self._record_state(
            delta_bid=0, delta_ask=0,
            n_buy=0, n_sell=0,
            w_t=0.0,  # Initial predator drift
            reward=0.0
        )

        obs = self._get_observation()
        info = self._get_info()

        return obs, info
    
    def step(
        self,
        action: Dict[str, np.ndarray],
        w_t: Optional[float] = None,  # Predator drift
        f_t: Optional[float] = None,  # Macro attack control
        g_t: Optional[float] = None,  # Macro defense control
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep.

        Args:
            action: Dict with 'delta_bid' and 'delta_ask' (half-spreads as fractions)
            w_t: Predator drift control (default 0.0 if not provided)
            f_t: Macro attack control (updates every macro_freq steps)
            g_t: Macro defense control (updates every macro_freq steps)

        Returns:
            observation: New observation
            reward: CARA utility
            terminated: Episode finished naturally
            truncated: Episode truncated (max steps)
            info: Additional information
        """
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")

        # Default predator drift to 0 if not provided
        if w_t is None:
            w_t = 0.0

        # Update macro controls if provided and at macro decision point
        if self.state.t % self.macro_freq == 0:
            if f_t is not None:
                self.macro_f = np.clip(f_t, 0.0, 1.0)
            if g_t is not None:
                self.macro_g = np.clip(g_t, 0.0, 1.0)
            # Update regime transition matrix
            self.btc_process.set_controls(self.macro_f, self.macro_g)

        # Extract MM action
        delta_bid = float(action['delta_bid'][0])
        delta_ask = float(action['delta_ask'][0])
        
        # Clip to valid range
        delta_bid = np.clip(delta_bid, 0.0, 0.1)
        delta_ask = np.clip(delta_ask, 0.0, 0.1)
        
        # Post quotes
        bid = self.state.S * (1 - delta_bid)
        ask = self.state.S * (1 + delta_ask)
        
        # Sample order arrivals
        n_buy, n_sell = self.order_generator.sample_arrivals(
            delta_bid=delta_bid,
            delta_ask=delta_ask,
            dt=self.dt,
        )
        
        # Update inventory and cash
        # Buy orders: customer buys from us at ask → we sell, q decreases, cash increases
        # Sell orders: customer sells to us at bid → we buy, q increases, cash decreases
        self.state.q += n_sell - n_buy
        self.state.cash += n_buy * ask - n_sell * bid
        
        # Clip inventory to limits
        if abs(self.state.q) > self.inventory_limit:
            # Force liquidation if exceeds limit (penalty)
            excess = abs(self.state.q) - self.inventory_limit
            liquidation_penalty = excess * self.state.S * 0.01  # 1% penalty per unit
            self.state.cash -= liquidation_penalty
            self.state.q = np.sign(self.state.q) * self.inventory_limit
        
        # Evolve price with predator drift: dS = w_t * dt + σ(regime) * sqrt(dt) * Z
        regime_sigma = self.btc_process.regime_params['sigma'][self.current_regime]
        dW = np.random.randn() * np.sqrt(self.dt)
        dS = w_t * self.dt + regime_sigma * self.state.S * dW

        self.state.S = max(self.state.S + dS, 1.0)  # Ensure price stays positive

        # Simulate regime switching (Poisson process)
        # Transition rate from current regime
        Q = self.btc_process.transition_matrix
        exit_rate = -Q[self.current_regime, self.current_regime]

        # Probability of jump in dt
        jump_prob = exit_rate * self.dt

        if np.random.rand() < jump_prob:
            # Jump occurs - switch to other regime (only 2 regimes)
            self.current_regime = 1 - self.current_regime

        # Update state regime
        self.state.regime = self.current_regime

        # Move to next timestep
        self.state.t += 1
        
        # Compute reward (CARA utility of current wealth)
        wealth = self.state.wealth()
        reward = float(self.utility(wealth))
        
        # Check termination
        terminated = False
        truncated = self.state.t >= self.max_steps
        
        # Terminal settlement
        if truncated:
            # Liquidate inventory (pay terminal cost)
            liquidation_value = self.state.q * self.state.S
            liquidation_cost = abs(liquidation_value) * self.terminal_liquidation_cost
            self.state.cash += liquidation_value - liquidation_cost
            self.state.q = 0
            
            # Final reward
            final_wealth = self.state.wealth()
            reward = float(self.utility(final_wealth))
        
        # Record
        self._record_state(delta_bid, delta_ask, n_buy, n_sell, w_t, reward)
        
        obs = self._get_observation()
        info = self._get_info()
        
        if self.render_mode == 'human':
            self.render()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        # Recent returns for volatility estimation (from history)
        if len(self.history['mid_price']) > 1:
            recent_prices = np.array(self.history['mid_price'][-11:])  # Last 11 prices
            recent_returns = np.diff(np.log(recent_prices))
            # Pad if needed
            if len(recent_returns) < 10:
                recent_returns = np.pad(
                    recent_returns,
                    (10 - len(recent_returns), 0),
                    mode='constant',
                    constant_values=0,
                )
            recent_returns = recent_returns[-10:]  # Keep last 10
        else:
            recent_returns = np.zeros(10)

        return {
            'mid_price': np.array([self.state.S], dtype=np.float32),
            'inventory': np.array([self.state.q], dtype=np.float32),
            'cash': np.array([self.state.cash], dtype=np.float32),
            'time_remaining': np.array([1.0 - self.state.t / self.max_steps], dtype=np.float32),
            'recent_returns': recent_returns.astype(np.float32),
            'wealth': np.array([self.state.wealth()], dtype=np.float32),
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info."""
        # Compute regime statistics from history
        regime_history = np.array(self.history['regime'])
        regime_stable_pct = np.mean(regime_history == 0) if len(regime_history) > 0 else 0.5

        return {
            'mid_price': self.state.S,
            'inventory': self.state.q,
            'cash': self.state.cash,
            'wealth': self.state.wealth(),
            'regime': self.state.regime,
            'timestep': self.state.t,
            'regime_stable_pct': regime_stable_pct,
            'macro_f': self.macro_f,
            'macro_g': self.macro_g,
        }
    
    def _record_state(
        self,
        delta_bid: float,
        delta_ask: float,
        n_buy: int,
        n_sell: int,
        w_t: float,
        reward: float,
    ):
        """Record state to history."""
        self.history['mid_price'].append(self.state.S)
        self.history['inventory'].append(self.state.q)
        self.history['cash'].append(self.state.cash)
        self.history['wealth'].append(self.state.wealth())
        self.history['bid'].append(self.state.S * (1 - delta_bid))
        self.history['ask'].append(self.state.S * (1 + delta_ask))
        self.history['spread'].append(delta_bid + delta_ask)
        self.history['n_buy'].append(n_buy)
        self.history['n_sell'].append(n_sell)
        self.history['regime'].append(self.state.regime)
        self.history['predator_drift'].append(w_t)
        self.history['macro_f'].append(self.macro_f)
        self.history['macro_g'].append(self.macro_g)
        self.history['reward'].append(reward)
    
    def render(self):
        """Render environment state."""
        if self.render_mode == 'human':
            wealth = self.state.wealth()
            print(f"[t={self.state.t:4d}] S=${self.state.S:8.2f} | q={self.state.q:+3.0f} | "
                  f"cash=${self.state.cash:10.2f} | wealth=${wealth:10.2f} | "
                  f"regime={self.state.regime}")
    
    def get_history_df(self):
        """Get history as pandas DataFrame (if pandas available)."""
        try:
            import pandas as pd
            return pd.DataFrame(self.history)
        except ImportError:
            warnings.warn("pandas not available, returning dict")
            return self.history


def make_market_making_env(
    S_0: float = 50000.0,
    gamma: float = 0.01,
    max_steps: int = 2880,  # 12 hours at 15s timestep
    dt_minutes: float = 0.25,  # 15 seconds
    inventory_limit: float = 10.0,
    lambda_0: float = 100.0 * 252,  # Annualized (100 orders/day * 252 days)
    kappa: float = 1.5,
    macro_freq: int = 120,  # Every 30 minutes
    xi: float = 0.01,
    macro_f: float = 0.5,
    macro_g: float = 0.5,
    # Regime parameters (optional, from calibration)
    sigma_stable: Optional[float] = None,
    sigma_volatile: Optional[float] = None,
    mu_stable: Optional[float] = None,
    mu_volatile: Optional[float] = None,
    base_transition_rate: Optional[float] = None,
    **kwargs
) -> MarketMakingEnv:
    """
    Factory function for market making environment.

    Args:
        S_0: Initial BTC price ($)
        gamma: CARA risk aversion
        max_steps: Episode length (timesteps), default 2880 = 12 hours at 15s
        dt_minutes: Timestep size (minutes), default 0.25 = 15 seconds
        inventory_limit: Max absolute inventory
        lambda_0: Base order arrival intensity (annualized)
        kappa: Spread sensitivity in order arrivals
        macro_freq: How often macro controls can change (in steps)
        xi: Predator drift cost coefficient
        macro_f: Initial attack control
        macro_g: Initial defense control
        **kwargs: Additional arguments to MarketMakingEnv

    Returns:
        Configured MarketMakingEnv

    Example:
        # 12-hour market making episode with 15s decisions
        env = make_market_making_env(
            S_0=50000.0,
            gamma=0.01,           # Moderate risk aversion
            max_steps=2880,       # 12 hours
            inventory_limit=10.0,
            xi=0.01,              # Predator cost
        )

        # Run episode with strategies
        obs, info = env.reset(seed=42)
        for t in range(env.max_steps):
            # MM action
            action = {'delta_bid': np.array([0.001]),
                      'delta_ask': np.array([0.001])}

            # Predator drift (e.g., w* = -ξγq from paper)
            w_t = -env.xi * env.gamma * obs['inventory'][0]

            # Macro controls (every 30 min)
            f_t = 0.5 if t % env.macro_freq == 0 else None
            g_t = 0.5 if t % env.macro_freq == 0 else None

            obs, reward, done, truncated, info = env.step(action, w_t, f_t, g_t)
            if done or truncated:
                break
    """
    # Create custom BTC process if regime parameters provided
    btc_process = None
    if any(p is not None for p in [sigma_stable, sigma_volatile, mu_stable, mu_volatile, base_transition_rate]):
        # Use calibrated parameters
        btc_process = create_default_btc_process(
            vol_low=sigma_stable if sigma_stable is not None else 0.40,
            vol_high=sigma_volatile if sigma_volatile is not None else 0.80,
            mu_low=mu_stable if mu_stable is not None else 0.10,
            mu_high=mu_volatile if mu_volatile is not None else -0.05,
            base_switch_rate=base_transition_rate if base_transition_rate is not None else 0.5,
        )

    return MarketMakingEnv(
        btc_process=btc_process,
        S_0=S_0,
        gamma=gamma,
        max_steps=max_steps,
        dt_minutes=dt_minutes,
        inventory_limit=inventory_limit,
        lambda_0=lambda_0,
        kappa=kappa,
        macro_freq=macro_freq,
        xi=xi,
        macro_f=macro_f,
        macro_g=macro_g,
        **kwargs
    )
