# BTC Market Making Calibration Framework

## Overview

Two-part calibration system for Avellaneda-Stoikov market making game parameters:
1. **Regime calibration**: Extract volatility regimes and drift from OHLCV data
2. **Microstructure calibration**: Estimate order arrival dynamics from tick data

---

## Components

### 1. ASCalibrationResult (Dataclass)

Complete set of calibrated parameters:

```python
@dataclass
class ASCalibrationResult:
    # Regime parameters
    sigma_stable: float           # Volatility in stable regime (annualized)
    sigma_volatile: float         # Volatility in volatile regime (annualized)
    mu_stable: float              # Drift in stable regime (annualized)
    mu_volatile: float            # Drift in volatile regime (annualized)
    base_transition_rate: float   # Regime switching rate (per day)

    # Predator bounds
    xi_estimate: float            # Estimated predator cost coefficient
    max_drift_bound: float        # Maximum observed |drift| for bounds

    # Microstructure parameters
    lambda_0: float               # Base order arrival intensity (annualized)
    kappa: float                  # Spread sensitivity parameter
    avg_trade_size: float         # Average trade size (BTC)
    trades_per_day: float         # Average number of trades per day

    # Metadata
    data_source: str
    calibration_period: str
    n_observations: int
```

**Methods**:
- `.to_dict()` - Convert to dictionary
- `.to_env_config()` - Export for `make_market_making_env()`
- `.summary()` - Generate formatted summary string

---

### 2. RegimeCalibrator

Calibrates 2-regime model from OHLCV data using K-means clustering on rolling volatility.

**Methodology**:
1. Compute log returns from Close prices
2. Calculate rolling volatility (default 20-period window)
3. K-means clustering (n=2) to identify high/low vol regimes
4. Estimate drift and volatility per regime
5. Compute transition rates from regime persistence

**Usage**:
```python
from calibration.physical.as_calibration import RegimeCalibrator

calibrator = RegimeCalibrator(
    rolling_window=20,      # Days for rolling vol
    annual_factor=252,      # Trading days per year
)

result = calibrator.calibrate(df_ohlcv, price_col='Close')

# Returns dict with:
# - sigma_stable, sigma_volatile
# - mu_stable, mu_volatile
# - base_transition_rate
# - regime_labels, regime_vols
```

**Parameters**:
- `rolling_window`: Window size for rolling volatility calculation
- `annual_factor`: Annualization factor (252 for daily, 252*24*12 for 5min, etc.)

**Output**:
- Stable regime (low vol): σ₀, μ₀
- Volatile regime (high vol): σ₁, μ₁
- Base transition rate: μ₀ (per day)

---

### 3. MicrostructureCalibrator

Calibrates order arrival parameters from tick-level trade data.

**Methodology**:
1. Compute inter-arrival times between trades
2. Estimate base arrival rate: λ₀ = 1 / mean(inter_arrival)
3. Estimate spread sensitivity κ from price distribution
4. Calculate average trade size and trades per day

**Usage**:
```python
from calibration.physical.as_calibration import MicrostructureCalibrator

calibrator = MicrostructureCalibrator(
    annual_factor=252 * 24 * 60 * 60,  # Seconds in trading year
)

result = calibrator.calibrate(
    df_tick,
    timestamp_col='timestamp',  # Unix seconds
    price_col='price',
    amount_col='amount',
    type_col='type',  # Optional: 'buy'/'sell'
)

# Returns dict with:
# - lambda_0 (annualized)
# - kappa
# - avg_trade_size
# - trades_per_day
# - median_spread
```

**Spread Sensitivity Estimation**:
- Uses rolling median as proxy for "true" mid-price
- Computes relative spreads: |price - mid| / mid
- Heuristic: κ ≈ 1 / median_spread
- Clamped to range [0.5, 10.0]

**Note**: Very high λ₀ values (100M+/year) indicate institutional HFT data. This is realistic for venues like Gemini but may need scaling for simulation.

---

### 4. Predator Cost Estimation

Estimated from regime volatility differential:

```python
xi_estimate = (sigma_volatile - sigma_stable) / gamma
xi_estimate = np.clip(xi_estimate, 0.001, 0.1)
```

**Rationale**: Predatory risk should be comparable to regime volatility difference, scaled by risk aversion.

---

### 5. Helper Functions

#### calibrate_from_csv_and_parquet()

Complete calibration from both OHLCV CSV and tick parquet files:

```python
from calibration.physical.as_calibration import calibrate_from_csv_and_parquet

result = calibrate_from_csv_and_parquet(
    ohlcv_path='btc_ohlcv_daily.csv',
    tick_path='gemini_btcusd_trades.parquet',
    gamma=0.01,
)

print(result.summary())

# Export for environment
env_config = result.to_env_config()
```

#### quick_calibrate_from_yfinance()

Quick calibration using only yfinance (no tick data):

```python
from calibration.physical.as_calibration import quick_calibrate_from_yfinance

result = quick_calibrate_from_yfinance(
    ticker='BTC-USD',
    start_date='2024-01-01',
    end_date='2025-01-01',
    interval='1d',  # '1m', '5m', '1h', '1d', etc.
    gamma=0.01,
    lambda_0_default=100.0 * 252,  # Default if no tick data
    kappa_default=1.5,
)
```

---

## Demo Results (2024 BTC Data)

### Regime Calibration

From 366 daily observations (2024-01-01 to 2024-12-31):

| Regime | Volatility (σ) | Drift (μ) |
|--------|----------------|-----------|
| **Stable** | 37.05% | +28.44% |
| **Volatile** | 56.27% | +114.04% |

- **Transition rate**: 0.037/day
- **Average holding time**: 27.4 days
- **Vol ratio**: 1.52x

### Microstructure Calibration

From Gemini BTC-USD tick data:

- **λ₀**: 496,815,709/year (institutional HFT level)
- **κ**: 10.0 (high spread sensitivity)
- **Avg trade size**: 0.0117 BTC (~$1,093 @ $93k)
- **Trades per day**: 503

### Predator Parameters

- **ξ**: 0.1000
- **Max drift bound**: ±22.21% (annualized)

---

## Integration with Environment

### Direct Use

```python
from market_making import make_market_making_env
from calibration.physical.as_calibration import calibrate_from_csv_and_parquet

# Calibrate
result = calibrate_from_csv_and_parquet(
    ohlcv_path='btc_ohlcv_daily.csv',
    tick_path='gemini_btcusd_trades.parquet',
    gamma=0.01,
)

# Create environment with calibrated parameters
env = make_market_making_env(
    S_0=93429.20,  # Latest BTC price from data
    gamma=0.01,
    kappa=result.kappa,
    xi=result.xi_estimate,
    sigma_stable=result.sigma_stable,
    sigma_volatile=result.sigma_volatile,
    mu_stable=result.mu_stable,
    mu_volatile=result.mu_volatile,
    base_transition_rate=result.base_transition_rate,
    lambda_0=result.lambda_0,
)
```

### Via Config Dictionary

```python
env_config = result.to_env_config()
env_config.update({
    'S_0': 93429.20,
    'max_steps': 2880,
    'dt_minutes': 0.25,
})

env = make_market_making_env(**env_config)
```

---

## File Structure

```
src/calibration/physical/
└── as_calibration.py              # Complete calibration framework
    ├── ASCalibrationResult        # Dataclass for results
    ├── RegimeCalibrator           # OHLCV → regimes
    ├── MicrostructureCalibrator   # Tick → λ₀, κ
    ├── calibrate_from_csv_and_parquet()
    └── quick_calibrate_from_yfinance()

notebooks/
└── gemini_btcusd_trades.parquet   # Example tick data

demo_calibration.py                # Complete demo script
```

---

## Data Requirements

### OHLCV Data (CSV)

**Required columns**:
- `Date` (index): Datetime
- `Close`: Close price

**Format**:
- CSV with datetime index
- Example: `btc_ohlcv_daily.csv`

**Frequency**: Daily or intraday (5m, 1h, etc.)

### Tick Data (Parquet)

**Required columns**:
- `timestamp`: Unix timestamp (seconds)
- `price`: Trade price (numeric or string)
- `amount`: Trade size (numeric or string)

**Optional**:
- `type`: 'buy' or 'sell'

**Format**: Parquet file
**Example**: `gemini_btcusd_trades.parquet`

---

## Calibration Best Practices

### 1. Data Frequency

- **Daily OHLCV**: Good for long-term regime identification (1 year+)
- **Hourly OHLCV**: Better granularity, limited by yfinance (730 days max)
- **5-min OHLCV**: Best for short-term regimes, yfinance limited to 60 days
- **Tick data**: Essential for accurate microstructure parameters

### 2. Regime Window

- `rolling_window=20`: Standard for daily data (monthly volatility)
- `rolling_window=100`: For hourly data (~4 days)
- `rolling_window=288`: For 5-min data (~1 day)

### 3. Annualization Factors

- Daily data: `annual_factor=252`
- Hourly data: `annual_factor=252*24`
- 5-min data: `annual_factor=252*24*12`
- Tick data: `annual_factor=252*24*60*60` (seconds)

### 4. Parameter Scaling

For simulation with 15s timesteps, consider scaling λ₀:

```python
# If calibrated λ₀ is too high (>100M/year):
dt_sim = 0.25 / (252 * 1440)  # 15s in annual units
expected_arrivals_per_step = lambda_0 * dt_sim

# If > 100 arrivals/step, scale down:
lambda_0_scaled = 100 / dt_sim  # Target 100 arrivals per step max
```

---

## Validation

### Regime Calibration

- **Vol ratio**: Should be 1.3x - 2.5x for crypto
- **Transition rate**: 0.01 - 0.1 per day typical for BTC
- **Drift**: Can be large for crypto (±50%+)

### Microstructure

- **λ₀**:
  - Retail: 1K - 100K/year
  - Institutional: 100K - 10M/year
  - HFT: 10M+ /year
- **κ**: 0.5 - 10.0 (higher = more sensitive to spreads)
- **Trade size**:
  - BTC: 0.001 - 1.0 BTC typical
  - Large block: 1.0+ BTC

### Predator Cost

- **ξ**: 0.001 - 0.1 typical
- Should scale with volatility difference

---

## Known Limitations

1. **yfinance time restrictions**:
   - 5m data: 60 days max
   - 1h data: 730 days max
   - Daily: unlimited

2. **K-means regime identification**:
   - Simple clustering, may not capture all market phases
   - Assumes exactly 2 regimes
   - Alternative: HMM or change-point detection

3. **Spread sensitivity estimation**:
   - Heuristic based on realized spreads
   - Assumes exponential decay: λ(δ) = λ₀ exp(-κδ)
   - Real market may have different functional form

4. **Annualization**:
   - Assumes constant trading intensity
   - Crypto trades 24/7, traditional markets don't
   - May need adjustment for market hours

---

## Future Enhancements

1. **Multi-regime calibration**: Support for 3+ regimes
2. **Time-varying parameters**: Drift and vol that change over time
3. **Alternative distributions**: Student-t, NIG for fat tails
4. **Order flow imbalance**: Calibrate buy/sell asymmetry
5. **Intraday seasonality**: Account for time-of-day patterns
6. **Model selection**: AIC/BIC for optimal regime count

---

## References

### Paper
- `double_game.tex` - Double-layer market making game formulation

### Code
- `src/market_making/market_making_env.py` - Environment implementation
- `src/processes/regime_switching_btc.py` - Regime-switching process
- `src/processes/poisson_orders.py` - Order arrival model

### Data Sources
- **yfinance**: Free OHLCV data for BTC-USD and other assets
- **Gemini**: Public tick data API (parquet format)
- **Alternative**: CCXT, Binance API, CryptoDataDownload

---

## Contact

**Author**: Yunian Pan
**Email**: yp1170@nyu.edu
**Institution**: New York University, Dept. of Electrical & Computer Engineering

---

## License

See `LICENSE` file in repository root.
