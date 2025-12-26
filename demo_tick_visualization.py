"""
Demo: Tick Data Visualization

Demonstrates tick data visualization utilities with different time ranges.

Author: Yunian Pan
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
from utils.tick_visualizer import visualize_tick_data, plot_price_microstructure

print("=" * 80)
print("TICK DATA VISUALIZATION DEMO")
print("=" * 80)

# Load data
df = pd.read_parquet('gemini_btcusd_full.parquet')

print(f"\nLoaded {len(df):,} trades")
print(f"Time range: {pd.to_datetime(df['timestamp'].min(), unit='s')} to {pd.to_datetime(df['timestamp'].max(), unit='s')}")

# ============================================================================
# Visualization 1: First Minute (High Activity)
# ============================================================================

print("\n" + "-" * 80)
print("VISUALIZATION 1: First Minute")
print("-" * 80)

start_ts = df['timestamp'].min()



print(f"✓ Created: tick_viz_first_minute.png")

# ============================================================================
# Visualization 2: Middle 30 Seconds (Detailed Microstructure)
# ============================================================================

print("\n" + "-" * 80)
print("VISUALIZATION 2: Middle 30 Seconds (Microstructure)")
print("-" * 80)

mid_ts = (df['timestamp'].min() + df['timestamp'].max()) / 2

fig2 = plot_price_microstructure(
    df,
    start_time=mid_ts,
    duration_seconds=30,
    save_path='tick_viz_microstructure_30s.png',
)

print(f"✓ Created: tick_viz_microstructure_30s.png")

# ============================================================================
# Visualization 3: 2-Minute Window (Balanced View)
# ============================================================================

print("\n" + "-" * 80)
print("VISUALIZATION 3: 2-Minute Window")
print("-" * 80)

# Start 1 minute from beginning
two_min_start = df['timestamp'].min() + 60

fig3 = visualize_tick_data(
    df,
    start_time=two_min_start,
    duration_seconds=120,
    title="Gemini BTC-USD: 2-Minute Trading Window",
    save_path='tick_viz_2min.png',
)

print(f"✓ Created: tick_viz_2min.png")

# ============================================================================
# Visualization 4: Full Range Overview (Aggregated)
# ============================================================================

print("\n" + "-" * 80)
print("VISUALIZATION 4: Full Range (~10 minutes)")
print("-" * 80)

# For full range, use all data
fig4 = visualize_tick_data(
    df,
    title="Gemini BTC-USD: Full Session Overview (50,000 trades)",
    save_path='tick_viz_full_range.png',
)

print(f"✓ Created: tick_viz_full_range.png")

# ============================================================================
# Summary and Recommendations
# ============================================================================

print("\n" + "=" * 80)
print("VISUALIZATION SUMMARY")
print("=" * 80)

print("""
RECOMMENDED TIME RANGES for Gemini BTC-USD data:

1. DETAILED MICROSTRUCTURE (10-30 seconds):
   - Best for: Studying individual trades, spread dynamics
   - Trades shown: 500-1,500
   - Use: plot_price_microstructure()

2. SHORT-TERM ACTIVITY (1-2 minutes):
   - Best for: Order flow patterns, volume clusters
   - Trades shown: 5,000-10,000
   - Use: visualize_tick_data(duration_seconds=60)

3. MEDIUM-TERM OVERVIEW (5 minutes):
   - Best for: Price trends, imbalance patterns
   - Trades shown: ~25,000
   - Use: visualize_tick_data(duration_seconds=300)

4. FULL SESSION (10 minutes):
   - Best for: Overall statistics, distribution analysis
   - Trades shown: 50,000
   - Use: visualize_tick_data() without time filters
   - Note: Individual ticks hard to see, but aggregated views work well

GENERAL GUIDELINES:
- For tick-by-tick analysis: < 1 minute (< 5,000 trades)
- For flow patterns: 1-5 minutes (5,000-25,000 trades)
- For statistical overview: Full range (50,000+ trades)
- High-frequency data (>5000 trades/min): Use 10-30s windows
""")

print("=" * 80)
print("✓ DEMO COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  - tick_viz_first_minute.png")
print("  - tick_viz_microstructure_30s.png")
print("  - tick_viz_2min.png")
print("  - tick_viz_full_range.png")
print("\nThese visualizations show trade-by-trade dynamics at different timescales.")
print("=" * 80 + "\n")
