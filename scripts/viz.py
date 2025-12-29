#!/usr/bin/env python3
"""
Trade Visualization Script

Displays trade details in an interactive Plotly chart from two sources:
1. TP/SL optimizer trade details
2. Main script backtest exports

Shows:
- 15-minute candles (transparent background layer)
- 1-minute candles (main layer)
- FVG zones on 15-minute timeframe (green=long, red=short)
- Entry and exit points
- TP/SL levels evolution during the trade
- Hover details for each bar

For backtests, additionally highlights:
- c1, c2, c3 pattern candles
- TP/SL adjustments over time

Usage:
    python visualize_trade.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
BACKTESTS_DIR = PROJECT_ROOT / "outputs" / "backtests"

sys.path.insert(0, str(PROJECT_ROOT))

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Error: plotly not installed. Run: pip install plotly")
    sys.exit(1)

# Import config for trade duration calculation
try:
    from config import get_trade_duration_minutes, ET
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("Warning: config module not found, trade window extensions will not be shown")


# =============================================================================
# CONFIGURATION
# =============================================================================

TRADING_PARAMS_DIR = PROJECT_ROOT / "outputs" / "optimization" / "trading_parameters"
CSV_PATTERN = "tpsl_trade_details_*.csv"
BACKTEST_PATTERN = "backtest*.csv"

# FVG entry window: number of 15-min candles the FVG remains valid
FVG_ENTRY_WINDOW_CANDLES = 6  # 6 x 15min = 90 minutes


# =============================================================================
# HELPER FUNCTION FOR HOVER-ENABLED LEGEND
# =============================================================================

def show_figure_with_hover_legend(fig: go.Figure):
    """
    Display a Plotly figure with a legend that appears on hover over a button.
    Uses a temporary HTML file with custom JavaScript to enable hover behavior.
    """
    import tempfile
    import webbrowser
    
    # Create temporary HTML file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    # Write figure to HTML with custom JavaScript for hover behavior
    html_str = fig.to_html(include_plotlyjs='cdn')
    
    # Inject custom JavaScript for hover behavior
    hover_script = """
    <script>
    (function() {
        let legendVisible = false;
        let plotlyDiv = null;
        let hoverButton = null;
        
        function showLegend() {
            if (!plotlyDiv) return;
            legendVisible = true;
            console.log('Showing legend');
            // Use Plotly's API to show legend
            Plotly.relayout(plotlyDiv, {'legend.visible': true}).catch(function(err) {
                console.error('Error showing legend:', err);
            });
        }
        
        function hideLegend() {
            if (!plotlyDiv) return;
            legendVisible = false;
            console.log('Hiding legend');
            // Use Plotly's API to hide legend
            Plotly.relayout(plotlyDiv, {'legend.visible': false}).catch(function(err) {
                console.error('Error hiding legend:', err);
            });
        }
        
        function toggleLegend() {
            console.log('Toggling legend, current state:', legendVisible);
            if (legendVisible) {
                hideLegend();
            } else {
                showLegend();
            }
        }
        
        function setupHoverLegend() {
            // Find all div elements that might be the Plotly graph
            const allDivs = document.querySelectorAll('div');
            console.log('Total divs found:', allDivs.length);
            
            // Look for the div that has Plotly data
            for (let div of allDivs) {
                if (div.data && div.layout) {
                    plotlyDiv = div;
                    console.log('Found Plotly div with data and layout');
                    break;
                }
            }
            
            // Fallback: look for div with class 'plotly-graph-div'
            if (!plotlyDiv) {
                plotlyDiv = document.querySelector('.plotly-graph-div');
                if (plotlyDiv) {
                    console.log('Found Plotly div with class plotly-graph-div');
                }
            }
            
            // Fallback: look for any div with js-plotly-plot class
            if (!plotlyDiv) {
                plotlyDiv = document.querySelector('.js-plotly-plot');
                if (plotlyDiv) {
                    console.log('Found Plotly div with class js-plotly-plot');
                }
            }
            
            if (!plotlyDiv) {
                console.error('Could not find Plotly div, retrying...');
                setTimeout(setupHoverLegend, 100);
                return;
            }
            
            // Wait for Plotly to be fully initialized
            if (typeof Plotly === 'undefined') {
                console.error('Plotly not loaded yet, retrying...');
                setTimeout(setupHoverLegend, 100);
                return;
            }
            
            console.log('Plotly is ready, setting up hover button');
            
            // Create a hover button overlay
            hoverButton = document.createElement('div');
            hoverButton.innerHTML = '🔑 Legend';
            hoverButton.style.position = 'absolute';
            hoverButton.style.top = '10px';
            hoverButton.style.left = '10px';
            hoverButton.style.width = 'auto';
            hoverButton.style.height = 'auto';
            hoverButton.style.display = 'flex';
            hoverButton.style.alignItems = 'center';
            hoverButton.style.justifyContent = 'center';
            hoverButton.style.backgroundColor = 'rgba(255, 255, 255, 0.95)';
            hoverButton.style.border = '2px solid rgba(0, 0, 0, 0.3)';
            hoverButton.style.borderRadius = '6px';
            hoverButton.style.cursor = 'pointer';
            hoverButton.style.zIndex = '10000';
            hoverButton.style.fontSize = '14px';
            hoverButton.style.padding = '6px 10px';
            hoverButton.style.boxSizing = 'border-box';
            hoverButton.style.userSelect = 'none';
            hoverButton.style.fontFamily = 'Arial, sans-serif';
            hoverButton.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
            hoverButton.title = 'Hover or click to show/hide legend';
            
            // Find the container
            let container = plotlyDiv.parentElement;
            if (!container || container === document.body) {
                container = document.body;
            }
            
            // Ensure container has relative positioning
            const containerStyle = window.getComputedStyle(container);
            if (containerStyle.position === 'static') {
                container.style.position = 'relative';
            }
            
            // Insert button
            container.insertBefore(hoverButton, plotlyDiv);
            console.log('Button created and inserted');
            
            // Initially hide the legend using Plotly API
            hideLegend();
            
            // Show legend on hover
            hoverButton.addEventListener('mouseenter', function() {
                console.log('Button mouseenter');
                showLegend();
            });
            
            // Hide legend when mouse leaves button
            let hideTimeout = null;
            hoverButton.addEventListener('mouseleave', function() {
                console.log('Button mouseleave');
                hideTimeout = setTimeout(function() {
                    hideLegend();
                }, 500);
            });
            
            // Click to toggle
            hoverButton.addEventListener('click', function(e) {
                console.log('Button clicked');
                e.stopPropagation();
                e.preventDefault();
                if (hideTimeout) {
                    clearTimeout(hideTimeout);
                    hideTimeout = null;
                }
                toggleLegend();
            });
            
            console.log('Setup complete!');
        }
        
        // Wait for Plotly to load and then setup
        function init() {
            console.log('Initializing...');
            if (typeof Plotly !== 'undefined') {
                console.log('Plotly loaded, waiting before setup...');
                // Wait a bit more for Plotly to fully initialize
                setTimeout(setupHoverLegend, 1500);
            } else {
                console.log('Plotly not loaded yet, retrying...');
                setTimeout(init, 100);
            }
        }
        
        // Start initialization
        if (document.readyState === 'loading') {
            console.log('Document still loading, waiting for load event');
            window.addEventListener('load', init);
        } else {
            console.log('Document ready, starting init');
            init();
        }
    })();
    </script>
    """
    
    # Insert the script before closing body tag
    html_str = html_str.replace('</body>', hover_script + '</body>')
    
    # Write to temporary file
    with open(temp_path, 'w', encoding='utf-8') as f:
        f.write(html_str)
    
    # Open in browser
    webbrowser.open(f'file://{temp_path}')
    
    # Note: File will be cleaned up on next script run or system restart
    # For production, you might want to use a more persistent location


# =============================================================================
# DATA LOADING - OPTIMIZER
# =============================================================================

def list_optimizer_csv_files() -> List[Path]:
    """List all trade detail CSV files in the trading parameters directory."""
    if not TRADING_PARAMS_DIR.exists():
        print(f"Error: Directory not found: {TRADING_PARAMS_DIR}")
        return []
    
    csv_files = list(TRADING_PARAMS_DIR.glob(CSV_PATTERN))
    return sorted(csv_files, reverse=True)


def load_trade_csv(csv_path: Path) -> pd.DataFrame:
    """Load a trade details CSV file (optimizer format)."""
    import pytz
    ET = pytz.timezone('America/New_York')
    
    df = pd.read_csv(csv_path)
    # Parse datetime columns (handle mixed timezones by parsing to UTC first)
    for col in ['entry_time', 'exit_time']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert(ET)
    return df


def get_trade_by_id(df: pd.DataFrame, trade_id: str) -> Optional[pd.Series]:
    """Get a specific trade by its ID."""
    mask = df['trade_id'] == trade_id
    if mask.sum() == 0:
        return None
    return df[mask].iloc[0]


# =============================================================================
# DATA LOADING - BACKTESTS
# =============================================================================

def list_backtest_csv_files() -> List[Path]:
    """List all backtest CSV files, sorted by modification time (newest first)."""
    if not BACKTESTS_DIR.exists():
        print(f"Error: Directory not found: {BACKTESTS_DIR}")
        return []
    
    csv_files = []
    for pattern in [BACKTEST_PATTERN]:
        csv_files.extend(BACKTESTS_DIR.glob(pattern))
    
    # Filter to only include actual backtest files (not training, sample, etc.)
    csv_files = [f for f in csv_files if f.name.startswith('backtest') and f.suffix == '.csv']
    
    # Sort by modification time, newest first
    csv_files = sorted(csv_files, key=lambda f: f.stat().st_mtime, reverse=True)
    return csv_files


def load_backtest_csv(csv_path: Path) -> pd.DataFrame:
    """Load a backtest CSV file."""
    import pytz
    ET = pytz.timezone('America/New_York')
    
    df = pd.read_csv(csv_path)
    
    # Convert backtest_date + time columns to full datetime
    if 'backtest_date' in df.columns:
        # Entry time
        if 'entry_time' in df.columns:
            df['entry_datetime'] = pd.to_datetime(
                df['backtest_date'].astype(str) + ' ' + df['entry_time'].astype(str),
                format='%Y-%m-%d %H:%M:%S',
                errors='coerce'
            )
            df['entry_datetime'] = df['entry_datetime'].dt.tz_localize(ET, ambiguous='NaT', nonexistent='NaT')
        
        # Exit time
        if 'exit_time' in df.columns:
            df['exit_datetime'] = pd.to_datetime(
                df['backtest_date'].astype(str) + ' ' + df['exit_time'].astype(str),
                format='%Y-%m-%d %H:%M:%S',
                errors='coerce'
            )
            df['exit_datetime'] = df['exit_datetime'].dt.tz_localize(ET, ambiguous='NaT', nonexistent='NaT')
        
        # Pattern candle timestamps (c1, c2, c3)
        for candle in ['c1', 'c2', 'c3']:
            col = f'{candle}_timestamp'
            if col in df.columns:
                df[f'{candle}_datetime'] = pd.to_datetime(df[col], errors='coerce')
                if df[f'{candle}_datetime'].dt.tz is None:
                    df[f'{candle}_datetime'] = df[f'{candle}_datetime'].dt.tz_localize(ET, ambiguous='NaT', nonexistent='NaT')
        
        # Sweep candle timestamp (used by Reversal Sweep and Liquidity Sweep patterns)
        if 'sweep_candle_timestamp' in df.columns:
            df['sweep_candle_datetime'] = pd.to_datetime(df['sweep_candle_timestamp'], errors='coerce')
            if df['sweep_candle_datetime'].dt.tz is None:
                df['sweep_candle_datetime'] = df['sweep_candle_datetime'].dt.tz_localize(ET, ambiguous='NaT', nonexistent='NaT')
        
        # Reversal-specific timestamps (for Reversal Sweep patterns)
        if 'invalidation_candle_timestamp' in df.columns:
            df['invalidation_candle_datetime'] = pd.to_datetime(df['invalidation_candle_timestamp'], errors='coerce')
            if df['invalidation_candle_datetime'].dt.tz is None:
                df['invalidation_candle_datetime'] = df['invalidation_candle_datetime'].dt.tz_localize(ET, ambiguous='NaT', nonexistent='NaT')
        
        if 'confirmation_time' in df.columns:
            df['confirmation_datetime'] = pd.to_datetime(df['confirmation_time'], errors='coerce')
            if df['confirmation_datetime'].dt.tz is None:
                df['confirmation_datetime'] = df['confirmation_datetime'].dt.tz_localize(ET, ambiguous='NaT', nonexistent='NaT')
        elif '_confirmation_time' in df.columns:
            df['confirmation_datetime'] = pd.to_datetime(df['_confirmation_time'], errors='coerce')
            if df['confirmation_datetime'].dt.tz is None:
                df['confirmation_datetime'] = df['confirmation_datetime'].dt.tz_localize(ET, ambiguous='NaT', nonexistent='NaT')
        
        # Confirmation start time (for reversal confirmation window visualization)
        if 'confirmation_start_time' in df.columns:
            df['confirmation_start_datetime'] = pd.to_datetime(df['confirmation_start_time'], errors='coerce')
            if df['confirmation_start_datetime'].dt.tz is None:
                df['confirmation_start_datetime'] = df['confirmation_start_datetime'].dt.tz_localize(ET, ambiguous='NaT', nonexistent='NaT')
    
    return df


def get_symbols_from_backtest(df: pd.DataFrame) -> List[str]:
    """Get unique symbols from backtest dataframe, sorted alphabetically."""
    if 'symbol' not in df.columns:
        return []
    return sorted(df['symbol'].unique().tolist())


def get_trades_for_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Get all trades for a specific symbol, sorted by c1_timestamp ascending."""
    mask = df['symbol'] == symbol
    trades = df[mask].copy()
    
    # Sort by c1 timestamp
    if 'c1_datetime' in trades.columns:
        trades = trades.sort_values('c1_datetime', ascending=True)
    elif 'c1_timestamp' in trades.columns:
        trades = trades.sort_values('c1_timestamp', ascending=True)
    
    return trades.reset_index(drop=True)


# =============================================================================
# PRICE DATA FETCHING
# =============================================================================

# Expected number of 1-minute bars in a full trading day (9:30 AM - 4:00 PM = 390 minutes)
EXPECTED_BARS_FULL_DAY = 390
# Minimum acceptable bars (at least 50% of expected)
MIN_ACCEPTABLE_BARS = 150
# Maximum allowed gap in minutes before considering data incomplete
MAX_GAP_MINUTES = 5

# API retry configuration
MAX_API_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0
BACKOFF_MULTIPLIER = 2.0


def check_bars_for_gaps(df: pd.DataFrame, max_gap_minutes: int = MAX_GAP_MINUTES) -> Tuple[bool, List[Tuple[datetime, datetime]]]:
    """
    Check if bar data has gaps larger than the allowed threshold.
    
    Args:
        df: DataFrame with 'timestamp' column
        max_gap_minutes: Maximum allowed gap in minutes
        
    Returns:
        Tuple of (has_gaps: bool, gaps: List of (start, end) tuples)
    """
    if df is None or df.empty or len(df) < 2:
        return False, []
    
    # Sort by timestamp
    sorted_df = df.sort_values('timestamp')
    
    gaps = []
    timestamps = sorted_df['timestamp'].tolist()
    
    for i in range(1, len(timestamps)):
        prev_ts = timestamps[i - 1]
        curr_ts = timestamps[i]
        
        # Handle timezone-aware timestamps
        if hasattr(prev_ts, 'tzinfo') and prev_ts.tzinfo is not None:
            gap_minutes = (curr_ts - prev_ts).total_seconds() / 60
        else:
            gap_minutes = (pd.Timestamp(curr_ts) - pd.Timestamp(prev_ts)).total_seconds() / 60
        
        if gap_minutes > max_gap_minutes:
            gaps.append((prev_ts, curr_ts, gap_minutes))
    
    return len(gaps) > 0, gaps


def report_bar_coverage(df: pd.DataFrame, market_open: datetime, market_close: datetime):
    """Report coverage statistics for bar data."""
    if df is None or df.empty:
        print("   [COVERAGE] No bars available")
        return
    
    expected_bars = int((market_close - market_open).total_seconds() / 60)
    actual_bars = len(df)
    coverage_pct = (actual_bars / expected_bars) * 100 if expected_bars > 0 else 0
    
    # Get actual time span
    first_ts = df['timestamp'].min()
    last_ts = df['timestamp'].max()
    
    print(f"   [COVERAGE] {actual_bars}/{expected_bars} bars ({coverage_pct:.1f}%)")
    print(f"              First: {first_ts}, Last: {last_ts}")
    
    # Check for gaps
    has_gaps, gaps = check_bars_for_gaps(df)
    if has_gaps:
        print(f"   [WARNING] Found {len(gaps)} gap(s) in bar data:")
        for start, end, gap_mins in gaps[:5]:  # Show first 5 gaps
            print(f"             {start} -> {end} ({gap_mins:.0f} min gap)")
        if len(gaps) > 5:
            print(f"             ... and {len(gaps) - 5} more gaps")


def fetch_bars_from_alpaca(
    symbol: str,
    market_open: datetime,
    market_close: datetime,
    max_retries: int = MAX_API_RETRIES
) -> Optional[pd.DataFrame]:
    """
    Fetch bars from Alpaca API with retry logic and rate limit handling.
    
    Args:
        symbol: Stock symbol
        market_open: Start time for bars
        market_close: End time for bars
        max_retries: Maximum number of retry attempts
        
    Returns:
        DataFrame with bars or None if failed
    """
    import time
    
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError as e:
        print(f"   [ERROR] Alpaca SDK not installed: {e}")
        return None
    
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("   [ERROR] Alpaca API credentials not set in environment variables")
        print("          Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        return None
    
    client = StockHistoricalDataClient(api_key, secret_key)
    
    backoff = INITIAL_BACKOFF_SECONDS
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"   [API] Fetching {symbol} bars (attempt {attempt}/{max_retries})...")
            
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=market_open,
                end=market_close
            )
            
            bars_data = client.get_stock_bars(request)
            
            if bars_data is None:
                print(f"   [WARNING] API returned None response")
                if attempt < max_retries:
                    print(f"   [RETRY] Backing off {backoff:.1f}s before retry...")
                    time.sleep(backoff)
                    backoff *= BACKOFF_MULTIPLIER
                    continue
                else:
                    print(f"   [ERROR] API returned no data after {max_retries} attempts")
                    return None
            
            # Check if symbol has data
            if symbol not in bars_data.data or not bars_data.data[symbol]:
                print(f"   [WARNING] No bars returned for {symbol} on this date")
                print(f"             (This could be a non-trading day, delisted symbol, or data gap)")
                return None
            
            # Convert to DataFrame
            records = []
            for bar in bars_data.data[symbol]:
                records.append({
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                })
            
            df = pd.DataFrame(records)
            
            if df.empty:
                print(f"   [WARNING] Empty DataFrame after parsing API response")
                return None
            
            # CRITICAL: Convert UTC timestamps to ET for proper display
            import pytz
            ET = pytz.timezone('America/New_York')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_convert(ET)
            else:
                # Assume UTC if no timezone info
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(ET)
            
            print(f"   [SUCCESS] Loaded {len(df)} bars from Alpaca API (converted to ET)")
            
            # Warn if fewer bars than expected
            if len(df) < MIN_ACCEPTABLE_BARS:
                print(f"   [WARNING] Only {len(df)} bars (expected ~{EXPECTED_BARS_FULL_DAY})")
                print(f"             Partial data may affect visualization")
            
            return df
            
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            # Check for rate limiting
            if 'rate' in error_str or '429' in error_str or 'too many' in error_str:
                print(f"   [RATE LIMIT] API rate limit hit, backing off {backoff:.1f}s...")
                time.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER
                continue
            
            # Check for authentication errors (don't retry)
            if 'auth' in error_str or '401' in error_str or '403' in error_str:
                print(f"   [ERROR] Authentication failed: {e}")
                print("          Check your ALPACA_API_KEY and ALPACA_SECRET_KEY")
                return None
            
            # Other errors - retry with backoff
            print(f"   [ERROR] API error on attempt {attempt}: {e}")
            if attempt < max_retries:
                print(f"   [RETRY] Backing off {backoff:.1f}s before retry...")
                time.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER
            else:
                print(f"   [FAILED] All {max_retries} API attempts failed")
                print(f"            Last error: {last_error}")
    
    return None


def fetch_bars_for_trade(
    symbol: str,
    entry_time: datetime,
    exit_time: datetime,
    timeframe: str = '1Min'
) -> Optional[pd.DataFrame]:
    """
    Fetch price bars for the trade period.
    
    Fetches from market open to end of day to show full context.
    First tries cache, then falls back to API with retry logic.
    Detects and reports gaps in bar data.
    """
    import pytz
    ET = pytz.timezone('America/New_York')
    
    # Ensure timezone-aware
    if entry_time.tzinfo is None:
        entry_time = ET.localize(entry_time)
    else:
        entry_time = entry_time.astimezone(ET)
    
    if exit_time.tzinfo is None:
        exit_time = ET.localize(exit_time)
    else:
        exit_time = exit_time.astimezone(ET)
    
    # Get trading day bounds (9:30 AM to 4:00 PM ET)
    trade_date = entry_time.date()
    market_open = ET.localize(datetime.combine(trade_date, datetime.strptime("09:30", "%H:%M").time()))
    market_close = ET.localize(datetime.combine(trade_date, datetime.strptime("16:00", "%H:%M").time()))
    
    print(f"   Fetching bars for {symbol} on {trade_date} ({market_open.strftime('%H:%M')} - {market_close.strftime('%H:%M')})")
    
    bars = None
    cache_had_data = False
    cache_has_gaps = False
    
    # =========================================================================
    # STEP 1: Try cache first
    # =========================================================================
    try:
        from data.data_cache import get_cache
        cache = get_cache()
        if cache and cache.enabled:
            print(f"   [CACHE] Checking cache for {symbol}...")
            bars = cache.get_bars(symbol, timeframe, market_open, market_close)
            
            if bars is not None and not bars.empty:
                cache_had_data = True
                print(f"   [CACHE] Found {len(bars)} bars in cache")
                
                # Ensure cache timestamps are in ET
                bars['timestamp'] = pd.to_datetime(bars['timestamp'])
                if bars['timestamp'].dt.tz is not None:
                    bars['timestamp'] = bars['timestamp'].dt.tz_convert(ET)
                else:
                    # Cache might store as UTC without tz info
                    try:
                        bars['timestamp'] = bars['timestamp'].dt.tz_localize('UTC').dt.tz_convert(ET)
                    except Exception:
                        # Already localized - convert
                        bars['timestamp'] = bars['timestamp'].dt.tz_convert(ET)
                
                # Check for gaps in cached data
                has_gaps, gaps = check_bars_for_gaps(bars)
                if has_gaps:
                    cache_has_gaps = True
                    print(f"   [CACHE] WARNING: Cache data has {len(gaps)} gap(s):")
                    for start, end, gap_mins in gaps[:3]:
                        print(f"           {start} -> {end} ({gap_mins:.0f} min)")
                    if len(gaps) > 3:
                        print(f"           ... and {len(gaps) - 3} more")
                    print(f"   [CACHE] Trying API to fill gaps...")
                elif len(bars) >= MIN_ACCEPTABLE_BARS:
                    print(f"   [SUCCESS] Using cached data ({len(bars)} bars, no gaps)")
                    report_bar_coverage(bars, market_open, market_close)
                    return bars
                else:
                    print(f"   [WARNING] Cache only has {len(bars)} bars (need {MIN_ACCEPTABLE_BARS}+)")
                    print(f"             Trying API for better data...")
            else:
                print(f"   [CACHE] No cached data found for {symbol} on {trade_date}")
        else:
            print(f"   [CACHE] Cache not available/enabled")
    except Exception as e:
        print(f"   [CACHE] Cache error: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # STEP 2: Fetch from API
    # =========================================================================
    api_bars = fetch_bars_from_alpaca(symbol, market_open, market_close)
    
    if api_bars is not None and not api_bars.empty:
        # Check API data for gaps too
        api_has_gaps, api_gaps = check_bars_for_gaps(api_bars)
        if api_has_gaps:
            print(f"   [API] WARNING: API data also has {len(api_gaps)} gap(s)")
        
        # If we have both cache and API data, try to merge them
        if cache_had_data and bars is not None and not bars.empty:
            # Merge data: combine unique timestamps from both sources
            print(f"   [MERGE] Attempting to merge cache ({len(bars)}) and API ({len(api_bars)}) data...")
            
            try:
                # Ensure both have consistent timestamp format
                cache_df = bars.copy()
                api_df = api_bars.copy()
                
                # Normalize timestamps
                cache_df['timestamp'] = pd.to_datetime(cache_df['timestamp'])
                api_df['timestamp'] = pd.to_datetime(api_df['timestamp'])
                
                # Remove timezone for comparison if present
                if cache_df['timestamp'].dt.tz is not None:
                    cache_df['timestamp'] = cache_df['timestamp'].dt.tz_convert('America/New_York')
                if api_df['timestamp'].dt.tz is not None:
                    api_df['timestamp'] = api_df['timestamp'].dt.tz_convert('America/New_York')
                
                # Combine and deduplicate, preferring API data for conflicts
                merged = pd.concat([cache_df, api_df], ignore_index=True)
                merged = merged.drop_duplicates(subset=['timestamp'], keep='last')
                merged = merged.sort_values('timestamp').reset_index(drop=True)
                
                print(f"   [MERGE] Result: {len(merged)} bars (was cache={len(bars)}, api={len(api_bars)})")
                
                # Check if merged data is better
                merged_has_gaps, merged_gaps = check_bars_for_gaps(merged)
                if merged_has_gaps:
                    print(f"   [MERGE] Merged data still has {len(merged_gaps)} gap(s)")
                    for start, end, gap_mins in merged_gaps[:3]:
                        print(f"           {start} -> {end} ({gap_mins:.0f} min)")
                else:
                    print(f"   [SUCCESS] Merged data has no gaps!")
                
                report_bar_coverage(merged, market_open, market_close)
                return merged
                
            except Exception as e:
                print(f"   [MERGE] Error merging data: {e}")
                # Fall through to use API data
        
        # Use API data if no merge needed or merge failed
        print(f"   [SUCCESS] Using API data ({len(api_bars)} bars)")
        report_bar_coverage(api_bars, market_open, market_close)
        return api_bars
    
    # =========================================================================
    # STEP 3: Fall back to cache if API failed but cache had some data
    # =========================================================================
    if cache_had_data and bars is not None and not bars.empty:
        print(f"   [FALLBACK] API failed, using cache data ({len(bars)} bars)")
        print(f"              Note: This data may have gaps")
        report_bar_coverage(bars, market_open, market_close)
        return bars
    
    # =========================================================================
    # STEP 4: Complete failure
    # =========================================================================
    print(f"   [ERROR] Could not fetch bars for {symbol} on {trade_date}")
    print(f"           - Cache: {'had data with gaps' if cache_has_gaps else ('partial data' if cache_had_data else 'no data')}")
    print(f"           - API: failed after retries")
    print(f"           Possible causes:")
    print(f"           - Non-trading day (holiday/weekend)")
    print(f"           - Symbol was not listed on this date")
    print(f"           - Data provider gap")
    print(f"           - API rate limiting")
    
    return None


# =============================================================================
# 15-MINUTE BAR AGGREGATION
# =============================================================================

def aggregate_to_5min(bars_1min: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 1-minute bars to 5-minute bars.
    
    Aligns to :00, :05, :10, :15, :20, :25, :30, :35, :40, :45, :50, :55 minute boundaries.
    """
    if bars_1min.empty:
        return pd.DataFrame()
    
    df = bars_1min.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create 5-minute period labels aligned to :00, :05, :10, etc.
    # Floor to 5-minute boundaries
    df['period_5m'] = df['timestamp'].dt.floor('5min')
    
    # Aggregate OHLCV
    bars_5min = df.groupby('period_5m').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()
    
    bars_5min.rename(columns={'period_5m': 'timestamp'}, inplace=True)
    
    return bars_5min


def aggregate_to_15min(bars_1min: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 1-minute bars to 15-minute bars.
    
    Aligns to :00, :15, :30, :45 minute boundaries.
    """
    if bars_1min.empty:
        return pd.DataFrame()
    
    df = bars_1min.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create 15-minute period labels aligned to :00, :15, :30, :45
    # Floor to 15-minute boundaries
    df['period_15m'] = df['timestamp'].dt.floor('15min')
    
    # Aggregate OHLCV
    bars_15min = df.groupby('period_15m').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()
    
    bars_15min.rename(columns={'period_15m': 'timestamp'}, inplace=True)
    
    return bars_15min


# =============================================================================
# FVG DETECTION ON 15-MINUTE TIMEFRAME
# =============================================================================

def detect_fvgs_15min(bars_15min: pd.DataFrame) -> List[Dict]:
    """
    Detect Fair Value Gaps on 15-minute bars.
    
    Returns list of FVG dictionaries with:
    - type: 'long' or 'short'
    - gap_low, gap_high: price boundaries of the gap
    - timestamp: when FVG formed (c3 timestamp)
    - entry_window_end: when FVG entry window expires
    """
    if len(bars_15min) < 3:
        return []
    
    fvgs = []
    
    for i in range(2, len(bars_15min)):
        c1 = bars_15min.iloc[i - 2]
        c2 = bars_15min.iloc[i - 1]
        c3 = bars_15min.iloc[i]
        
        # Long FVG: gap between c1 high and c3 low
        # Bullish FVG = c3 low > c1 high (gap up)
        if c3['low'] > c1['high']:
            gap_low = c1['high']
            gap_high = c3['low']
            gap_size = gap_high - gap_low
            
            # Only include meaningful gaps (> 0.1% of price)
            if gap_size / c2['close'] > 0.001:
                fvgs.append({
                    'type': 'long',
                    'gap_low': gap_low,
                    'gap_high': gap_high,
                    'gap_size': gap_size,
                    'timestamp': c3['timestamp'],
                    'c2_high': c2['high'],
                    'c2_low': c2['low'],
                    'entry_window_end': c3['timestamp'] + timedelta(minutes=15 * FVG_ENTRY_WINDOW_CANDLES)
                })
        
        # Short FVG: gap between c3 high and c1 low
        # Bearish FVG = c3 high < c1 low (gap down)
        if c3['high'] < c1['low']:
            gap_low = c3['high']
            gap_high = c1['low']
            gap_size = gap_high - gap_low
            
            if gap_size / c2['close'] > 0.001:
                fvgs.append({
                    'type': 'short',
                    'gap_low': gap_low,
                    'gap_high': gap_high,
                    'gap_size': gap_size,
                    'timestamp': c3['timestamp'],
                    'c2_high': c2['high'],
                    'c2_low': c2['low'],
                    'entry_window_end': c3['timestamp'] + timedelta(minutes=15 * FVG_ENTRY_WINDOW_CANDLES)
                })
    
    return fvgs


# =============================================================================
# TP/SL LEVEL CALCULATION - OPTIMIZER
# =============================================================================

def calculate_tpsl_levels(
    trade: pd.Series,
    bars: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate TP/SL levels at each bar during the trade (for optimizer trades).
    
    Simulates the trailing stop and TP extension logic following optimizer rules.
    """
    entry_time = trade['entry_time']
    exit_time = trade['exit_time']
    direction = trade['direction']
    entry_price = trade['entry_price']
    
    # Initial levels from optimizer
    initial_stop = trade['initial_stop']
    initial_tp1 = trade['initial_tp1']
    initial_tp2 = trade.get('initial_tp2', initial_tp1)
    
    # Final levels from optimizer (what actually happened)
    final_stop = trade.get('final_stop', initial_stop)
    final_tp = trade.get('final_tp', initial_tp2)
    
    # Strategy parameters
    trailing_enabled = trade.get('trailing_enabled', False)
    trailing_activation_r = trade.get('trailing_activation_r', 1.0)
    trailing_atr_mult = trade.get('trailing_atr_mult', 0.5)
    rsi_extension_enabled = trade.get('rsi_extension_enabled', False)
    rsi_threshold = trade.get('rsi_threshold', 60 if direction == 'long' else 40)
    
    # Track adjustments
    stop_adjustments = 0
    tp_extensions = 0
    max_stop_adjustments = 20
    max_tp_extensions = 30
    
    # Calculate risk (R)
    risk = abs(entry_price - initial_stop)
    
    # Filter bars to trade period
    mask = (bars['timestamp'] >= entry_time) & (bars['timestamp'] <= exit_time)
    trade_bars = bars[mask].copy()
    
    if trade_bars.empty:
        return pd.DataFrame()
    
    # Initialize levels
    current_stop = initial_stop
    current_tp = initial_tp2  # Main TP after partial
    levels = []
    
    # Calculate ATR (14-period average true range approximation)
    # Use bars before entry for a proper ATR
    pre_entry_bars = bars[bars['timestamp'] < entry_time].tail(14)
    if len(pre_entry_bars) >= 5:
        tr = pre_entry_bars['high'] - pre_entry_bars['low']
        atr = tr.mean()
    else:
        atr = risk * 0.5  # Fallback
    
    if pd.isna(atr) or atr <= 0:
        atr = risk * 0.5
    
    # Calculate simple RSI for TP extension (approximation)
    def calc_rsi(prices, period=14):
        """Simple RSI calculation."""
        if len(prices) < period + 1:
            return 50  # Neutral
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50
    
    for idx, bar in trade_bars.iterrows():
        bar_time = bar['timestamp']
        bar_close = bar['close']
        bar_high = bar['high']
        bar_low = bar['low']
        
        # Get price history up to this bar for RSI
        history = bars[bars['timestamp'] <= bar_time]['close']
        current_rsi = calc_rsi(history)
        
        # =====================================================================
        # TRAILING STOP LOGIC
        # =====================================================================
        if trailing_enabled and risk > 0 and stop_adjustments < max_stop_adjustments:
            activation_move = risk * trailing_activation_r
            
            if direction == 'long':
                # Activate when price is (activation_r * risk) above entry
                if bar_close > entry_price + activation_move:
                    # Trail stop: ATR * mult below current close
                    new_stop = bar_close - (atr * trailing_atr_mult)
                    new_stop = round(new_stop, 2)
                    
                    # Only trail UP for long positions
                    if new_stop > current_stop:
                        current_stop = new_stop
                        stop_adjustments += 1
            else:  # short
                # Activate when price is (activation_r * risk) below entry
                if bar_close < entry_price - activation_move:
                    # Trail stop: ATR * mult above current close
                    new_stop = bar_close + (atr * trailing_atr_mult)
                    new_stop = round(new_stop, 2)
                    
                    # Only trail DOWN for short positions
                    if new_stop < current_stop:
                        current_stop = new_stop
                        stop_adjustments += 1
        
        # =====================================================================
        # RSI TP EXTENSION LOGIC
        # =====================================================================
        if rsi_extension_enabled and tp_extensions < max_tp_extensions:
            should_extend = False
            
            if direction == 'long':
                # Extend TP when RSI > threshold (bullish momentum)
                if current_rsi > rsi_threshold:
                    should_extend = True
            else:  # short
                # Extend TP when RSI < threshold (bearish momentum)
                if current_rsi < rsi_threshold:
                    should_extend = True
            
            if should_extend:
                tp_extension_amount = atr * 0.5  # Default extension
                if direction == 'long':
                    new_tp = current_tp + tp_extension_amount
                else:
                    new_tp = current_tp - tp_extension_amount
                
                current_tp = round(new_tp, 2)
                tp_extensions += 1
        
        levels.append({
            'timestamp': bar_time,
            'stop_level': current_stop,
            'tp_level': current_tp,
            'bar_close': bar_close,
            'rsi': current_rsi,
            'stop_adjustments': stop_adjustments,
            'tp_extensions': tp_extensions
        })
    
    return pd.DataFrame(levels)


# =============================================================================
# TP/SL LEVEL CALCULATION - BACKTEST
# =============================================================================

def parse_adjustment_list(times_str, values_str, backtest_date) -> Tuple[List[datetime], List[float]]:
    """Parse comma-separated adjustment times and values."""
    import pytz
    ET = pytz.timezone('America/New_York')
    
    times = []
    values = []
    
    if pd.isna(times_str) or pd.isna(values_str) or str(times_str).strip() == '' or str(values_str).strip() == '':
        return times, values
    
    try:
        times_list = str(times_str).split(',')
        values_list = str(values_str).split(',')
        
        for t, v in zip(times_list, values_list):
            t = t.strip()
            v = v.strip()
            if t and v:
                try:
                    # Parse time and combine with date
                    dt = pd.to_datetime(f"{backtest_date} {t}", format='%Y-%m-%d %H:%M:%S', errors='coerce')
                    if pd.isna(dt):
                        dt = pd.to_datetime(f"{backtest_date} {t}")
                    if dt.tzinfo is None:
                        dt = ET.localize(dt)
                    times.append(dt)
                    values.append(float(v))
                except:
                    pass
    except:
        pass
    
    return times, values


def calculate_tpsl_levels_backtest(
    trade: pd.Series,
    bars: pd.DataFrame
) -> Tuple[pd.DataFrame, List[Tuple[datetime, float]], List[Tuple[datetime, float]]]:
    """
    Calculate TP/SL levels at each bar during the trade from backtest data.
    
    Uses the stop_adjustment_times/values and tp_adjustment_times/values columns.
    
    Returns:
        - DataFrame with timestamp, stop_level, tp_level for each bar
        - List of (time, value) tuples for stop adjustments
        - List of (time, value) tuples for TP adjustments
    """
    import pytz
    ET = pytz.timezone('America/New_York')
    
    entry_time = trade.get('entry_datetime')
    exit_time = trade.get('exit_datetime')
    
    if pd.isna(entry_time) or pd.isna(exit_time):
        return pd.DataFrame(), [], []
    
    backtest_date = trade.get('backtest_date')
    
    # Get initial and final levels
    initial_stop = trade.get('initial_stop_loss', trade.get('original_stop_loss_post_readjust'))
    initial_tp = trade.get('initial_take_profit', trade.get('original_take_profit_post_readjust'))
    final_stop = trade.get('final_stop_loss', trade.get('stop_loss', initial_stop))
    final_tp = trade.get('final_take_profit', trade.get('take_profit', initial_tp))
    
    if pd.isna(initial_stop) or pd.isna(initial_tp):
        initial_stop = trade.get('stop_loss')
        initial_tp = trade.get('take_profit')
    
    # Parse adjustment times and values using the helper function
    stop_adj_times, stop_adj_values = parse_adjustment_list(
        trade.get('stop_adjustment_times', ''),
        trade.get('stop_adjustment_values', ''),
        backtest_date
    )
    
    tp_adj_times, tp_adj_values = parse_adjustment_list(
        trade.get('tp_adjustment_times', ''),
        trade.get('tp_adjustment_values', ''),
        backtest_date
    )
    
    # Create adjustment point lists for drawing horizontal lines
    stop_adjustments_list = list(zip(stop_adj_times, stop_adj_values))
    tp_adjustments_list = list(zip(tp_adj_times, tp_adj_values))
    
    # Filter bars to trade period
    mask = (bars['timestamp'] >= entry_time) & (bars['timestamp'] <= exit_time)
    trade_bars = bars[mask].copy()
    
    if trade_bars.empty:
        return pd.DataFrame(), stop_adjustments_list, tp_adjustments_list
    
    # Build levels list
    levels = []
    current_stop = initial_stop if not pd.isna(initial_stop) else 0
    current_tp = initial_tp if not pd.isna(initial_tp) else 0
    
    stop_adj_idx = 0
    tp_adj_idx = 0
    
    for idx, bar in trade_bars.iterrows():
        bar_time = bar['timestamp']
        
        # Apply stop adjustments up to this bar's time
        while stop_adj_idx < len(stop_adj_times) and stop_adj_times[stop_adj_idx] <= bar_time:
            current_stop = stop_adj_values[stop_adj_idx]
            stop_adj_idx += 1
        
        # Apply TP adjustments up to this bar's time
        while tp_adj_idx < len(tp_adj_times) and tp_adj_times[tp_adj_idx] <= bar_time:
            current_tp = tp_adj_values[tp_adj_idx]
            tp_adj_idx += 1
        
        levels.append({
            'timestamp': bar_time,
            'stop_level': current_stop,
            'tp_level': current_tp,
            'bar_close': bar['close'],
            'stop_adjustments': stop_adj_idx,
            'tp_extensions': tp_adj_idx
        })
    
    return pd.DataFrame(levels), stop_adjustments_list, tp_adjustments_list


# =============================================================================
# VISUALIZATION - OPTIMIZER
# =============================================================================

def create_trade_chart(
    trade: pd.Series,
    bars_1min: pd.DataFrame,
    bars_15min: pd.DataFrame,
    fvgs: List[Dict],
    tpsl_levels: pd.DataFrame
) -> go.Figure:
    """Create an interactive Plotly chart for the trade (optimizer format)."""
    
    symbol = trade['symbol']
    direction = trade['direction']
    entry_time = trade['entry_time']
    exit_time = trade['exit_time']
    entry_price = trade['entry_price']
    exit_price = trade['exit_price']
    exit_reason = trade['exit_reason']
    simulated_pnl = trade.get('simulated_pnl', 0)
    
    # Create figure
    fig = go.Figure()
    
    # =========================================================================
    # LAYER 1 (BACKGROUND): FVG ZONES
    # =========================================================================
    for i, fvg in enumerate(fvgs):
        fvg_color = 'rgba(76, 175, 80, 0.15)' if fvg['type'] == 'long' else 'rgba(244, 67, 54, 0.15)'
        border_color = 'rgba(76, 175, 80, 0.5)' if fvg['type'] == 'long' else 'rgba(244, 67, 54, 0.5)'
        
        fig.add_shape(
            type="rect",
            x0=fvg['timestamp'],
            x1=fvg['entry_window_end'],
            y0=fvg['gap_low'],
            y1=fvg['gap_high'],
            fillcolor=fvg_color,
            line=dict(color=border_color, width=1),
            layer="below",
            name=f"FVG {fvg['type']}"
        )
        
        # Add FVG label (invisible scatter for hover)
        fig.add_trace(go.Scatter(
            x=[fvg['timestamp'] + timedelta(minutes=7)],
            y=[(fvg['gap_low'] + fvg['gap_high']) / 2],
            mode='markers',
            marker=dict(size=1, opacity=0),
            showlegend=False,
            hovertemplate=(
                f"<b>FVG ({fvg['type'].upper()})</b><br>"
                f"Gap: ${fvg['gap_low']:.2f} - ${fvg['gap_high']:.2f}<br>"
                f"Size: ${fvg['gap_size']:.2f}<br>"
                f"Valid until: {fvg['entry_window_end'].strftime('%H:%M')}<br>"
                f"<extra></extra>"
            )
        ))
    
    # =========================================================================
    # LAYER 2: 15-MINUTE CANDLES (TRANSPARENT BACKGROUND)
    # Draw as shapes to control width (span full 15 minutes)
    # =========================================================================
    for _, bar15 in bars_15min.iterrows():
        is_bullish = bar15['close'] >= bar15['open']
        
        # Colors: more opaque than before
        if is_bullish:
            fill_color = 'rgba(38, 166, 154, 0.35)'
            line_color = 'rgba(38, 166, 154, 0.6)'
        else:
            fill_color = 'rgba(239, 83, 80, 0.35)'
            line_color = 'rgba(239, 83, 80, 0.6)'
        
        bar_start = bar15['timestamp']
        bar_end = bar_start + timedelta(minutes=15)
        
        # Draw candle body (rectangle from open to close)
        body_low = min(bar15['open'], bar15['close'])
        body_high = max(bar15['open'], bar15['close'])
        
        fig.add_shape(
            type="rect",
            x0=bar_start,
            x1=bar_end,
            y0=body_low,
            y1=body_high,
            fillcolor=fill_color,
            line=dict(color=line_color, width=1),
            layer="below",
            xref='x',
            yref='y'
        )
        
        # Draw wicks (vertical lines for high/low)
        wick_x = bar_start + timedelta(minutes=7.5)  # Center of the bar
        
        # Upper wick
        if bar15['high'] > body_high:
            fig.add_shape(
                type="line",
                x0=wick_x,
                x1=wick_x,
                y0=body_high,
                y1=bar15['high'],
                line=dict(color=line_color, width=1),
                layer="below",
                xref='x',
                yref='y'
            )
        
        # Lower wick
        if bar15['low'] < body_low:
            fig.add_shape(
                type="line",
                x0=wick_x,
                x1=wick_x,
                y0=bar15['low'],
                y1=body_low,
                line=dict(color=line_color, width=1),
                layer="below",
                xref='x',
                yref='y'
            )
    
    # Add invisible scatter for 15-min hover info
    if not bars_15min.empty:
        fig.add_trace(go.Scatter(
            x=bars_15min['timestamp'] + pd.Timedelta(minutes=7.5),
            y=(bars_15min['high'] + bars_15min['low']) / 2,
            mode='markers',
            marker=dict(size=1, opacity=0),
            name='15min',
            showlegend=True,
            hovertemplate=[
                f"<b>15min Bar</b><br>Time: {t.strftime('%H:%M')}<br>"
                f"O: ${o:.2f}<br>H: ${h:.2f}<br>L: ${l:.2f}<br>C: ${c:.2f}<extra></extra>"
                for t, o, h, l, c in zip(
                    bars_15min['timestamp'], bars_15min['open'], bars_15min['high'],
                    bars_15min['low'], bars_15min['close']
                )
            ]
        ))
    
    # =========================================================================
    # LAYER 3: 1-MINUTE CANDLES (MAIN LAYER)
    # =========================================================================
    fig.add_trace(go.Candlestick(
        x=bars_1min['timestamp'],
        open=bars_1min['open'],
        high=bars_1min['high'],
        low=bars_1min['low'],
        close=bars_1min['close'],
        name='1min',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        hoverinfo='text',
        hovertext=[
            f"Time: {t}<br>O: ${o:.2f}<br>H: ${h:.2f}<br>L: ${l:.2f}<br>C: ${c:.2f}<br>Vol: {v:,.0f}"
            for t, o, h, l, c, v in zip(
                bars_1min['timestamp'], bars_1min['open'], bars_1min['high'], 
                bars_1min['low'], bars_1min['close'], bars_1min['volume']
            )
        ]
    ))
    
    # =========================================================================
    # LAYER 4: ENTRY AND EXIT MARKERS
    # =========================================================================
    
    # Entry marker
    entry_color = '#2196F3' if direction == 'long' else '#FF9800'
    entry_symbol = 'triangle-up' if direction == 'long' else 'triangle-down'
    
    fig.add_trace(go.Scatter(
        x=[entry_time],
        y=[entry_price],
        mode='markers',
        marker=dict(
            size=15,
            color=entry_color,
            symbol=entry_symbol,
            line=dict(width=2, color='black')
        ),
        name='Entry',
        hovertemplate=(
            f"<b>ENTRY ({direction.upper()})</b><br>"
            f"Time: {entry_time}<br>"
            f"Price: ${entry_price:.2f}<br>"
            f"<extra></extra>"
        )
    ))
    
    # Exit marker
    exit_color = '#4CAF50' if simulated_pnl > 0 else '#F44336'
    
    fig.add_trace(go.Scatter(
        x=[exit_time],
        y=[exit_price],
        mode='markers',
        marker=dict(
            size=15,
            color=exit_color,
            symbol='x',
            line=dict(width=2, color='black')
        ),
        name='Exit',
        hovertemplate=(
            f"<b>EXIT ({exit_reason})</b><br>"
            f"Time: {exit_time}<br>"
            f"Price: ${exit_price:.2f}<br>"
            f"P&L: ${simulated_pnl:.2f}<br>"
            f"<extra></extra>"
        )
    ))
    
    # =========================================================================
    # LAYER 5: TP/SL LEVELS DURING TRADE
    # =========================================================================
    if not tpsl_levels.empty:
        # Stop loss line
        fig.add_trace(go.Scatter(
            x=tpsl_levels['timestamp'],
            y=tpsl_levels['stop_level'],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Stop Loss',
            hovertemplate="SL: $%{y:.2f}<extra></extra>"
        ))
        
        # Take profit line
        fig.add_trace(go.Scatter(
            x=tpsl_levels['timestamp'],
            y=tpsl_levels['tp_level'],
            mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name='Take Profit',
            hovertemplate="TP: $%{y:.2f}<extra></extra>"
        ))
    
    # =========================================================================
    # REFERENCE LINES
    # =========================================================================
    
    initial_stop = trade['initial_stop']
    initial_tp1 = trade['initial_tp1']
    initial_tp2 = trade.get('initial_tp2', initial_tp1)
    final_stop = trade.get('final_stop', initial_stop)
    final_tp = trade.get('final_tp', initial_tp2)
    
    # Count adjustments from CSV
    stop_adjustments = trade.get('stop_adjustments', 0)
    tp_extensions = trade.get('tp_extensions', 0)
    
    # Entry price line
    fig.add_hline(
        y=entry_price,
        line_dash="solid",
        line_color="rgba(33, 150, 243, 0.5)",
        annotation_text=f"Entry: ${entry_price:.2f}",
        annotation_position="left",
        row=1, col=1
    )
    
    # Initial stop line (dotted, faded)
    fig.add_hline(
        y=initial_stop,
        line_dash="dot",
        line_color="rgba(255, 0, 0, 0.25)",
        annotation_text=f"Init SL: ${initial_stop:.2f}",
        annotation_position="right",
        row=1, col=1
    )
    
    # Initial TP1 line (partial exit target)
    fig.add_hline(
        y=initial_tp1,
        line_dash="dot",
        line_color="rgba(0, 200, 0, 0.25)",
        annotation_text=f"TP1: ${initial_tp1:.2f}",
        annotation_position="right",
        row=1, col=1
    )
    
    # Initial TP2 line (runner target) - if different from TP1
    if abs(initial_tp2 - initial_tp1) > 0.01:
        fig.add_hline(
            y=initial_tp2,
            line_dash="dot",
            line_color="rgba(0, 150, 0, 0.25)",
            annotation_text=f"TP2: ${initial_tp2:.2f}",
            annotation_position="right",
            row=1, col=1
        )
    
    # Final stop line (if different from initial - shows trailing effect)
    if abs(final_stop - initial_stop) > 0.01:
        fig.add_hline(
            y=final_stop,
            line_dash="dashdot",
            line_color="rgba(255, 100, 0, 0.5)",
            annotation_text=f"Final SL: ${final_stop:.2f} ({stop_adjustments} adj)",
            annotation_position="right",
            row=1, col=1
        )
    
    # Final TP line (if different from initial - shows extension effect)
    if abs(final_tp - initial_tp2) > 0.01:
        fig.add_hline(
            y=final_tp,
            line_dash="dashdot",
            line_color="rgba(0, 200, 100, 0.5)",
            annotation_text=f"Final TP: ${final_tp:.2f} ({tp_extensions} ext)",
            annotation_position="right",
            row=1, col=1
        )
    
    # Shade trade period - with extension visualization
    # Calculate original trade window end based on max_duration
    has_adjustments = stop_adjustments > 0 or tp_extensions > 0
    
    if CONFIG_AVAILABLE and has_adjustments:
        # Get max duration for this trade's direction
        trading_tf = trade.get('trading_timeframe', '15min')
        max_duration = get_trade_duration_minutes(direction, trading_tf)
        original_window_end = entry_time + timedelta(minutes=max_duration)
        
        # If trade extended beyond original window, show both regions
        if exit_time > original_window_end:
            # Initial window (light gray) - only on main chart
            fig.add_vrect(
                x0=entry_time,
                x1=original_window_end,
                fillcolor="rgba(100, 100, 100, 0.08)",
                line_width=0,
                annotation_text="Initial Window",
                annotation_position="top left",
                row=1, col=1
            )
            # Extended window (light blue) - only on main chart
            fig.add_vrect(
                x0=original_window_end,
                x1=exit_time,
                fillcolor="rgba(100, 150, 255, 0.12)",
                line_width=0,
                annotation_text="Extended",
                annotation_position="top left",
                row=1, col=1
            )
        else:
            # Trade ended within original window - only on main chart
            fig.add_vrect(
                x0=entry_time,
                x1=exit_time,
                fillcolor="rgba(100, 100, 100, 0.08)",
                line_width=0,
                annotation_text="Trade Period",
                annotation_position="top left",
                row=1, col=1
            )
    else:
        # No adjustments or config not available - single gray region - only on main chart
        fig.add_vrect(
            x0=entry_time,
            x1=exit_time,
            fillcolor="rgba(100, 100, 100, 0.08)",
            line_width=0,
            annotation_text="Trade Period",
            annotation_position="top left",
            row=1, col=1
        )
    
    # =========================================================================
    # LAYOUT
    # =========================================================================
    pnl_sign = '+' if simulated_pnl > 0 else ''
    long_fvgs = sum(1 for f in fvgs if f['type'] == 'long')
    short_fvgs = sum(1 for f in fvgs if f['type'] == 'short')
    
    # Strategy info
    trailing_enabled = trade.get('trailing_enabled', False)
    rsi_extension_enabled = trade.get('rsi_extension_enabled', False)
    trailing_info = "Trail: ON" if trailing_enabled else "Trail: OFF"
    rsi_info = "RSI Ext: ON" if rsi_extension_enabled else "RSI Ext: OFF"
    
    # Format timestamp for title
    pattern_type = trade.get('pattern_type', 'FVG')
    entry_time_str = ""
    if entry_time is not None:
        if hasattr(entry_time, 'strftime'):
            entry_time_str = entry_time.strftime('%Y-%m-%d %H:%M')
        else:
            entry_time_str = str(entry_time)
    
    # Title format: timestamp | symbol | direction | pattern_type
    title = (
        f"<b>{entry_time_str} | {symbol} | {direction.upper()} | {pattern_type}</b><br>"
        f"<span style='font-size:14px'>Entry: ${entry_price:.2f} -> Exit: ${exit_price:.2f} "
        f"({exit_reason}) | P&L: {pnl_sign}${simulated_pnl:.2f}</span><br>"
        f"<span style='font-size:12px'>{trailing_info} | {rsi_info} | "
        f"SL adj: {stop_adjustments} | TP ext: {tp_extensions} | "
        f"FVGs: {long_fvgs}L/{short_fvgs}S</span>"
    )
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Time",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        height=750,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.9)",
            visible=False  # Hidden by default, shown on hover
        ),
        hovermode='x unified',
        annotations=[
            dict(
                text="🔑",
                xref="paper", yref="paper",
                x=0.01, y=0.99,
                xanchor="left", yanchor="top",
                showarrow=False,
                font=dict(size=16),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.3)",
                borderwidth=1,
                borderpad=4
            )
        ]
    )
    
    return fig


# =============================================================================
# VISUALIZATION - BACKTEST
# =============================================================================

def calculate_rsi_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate RSI for all bars in DataFrame.
    
    Args:
        df: DataFrame with 'close' column
        period: RSI period (default 14)
    
    Returns:
        Series of RSI values
    """
    if len(df) < period + 1:
        return pd.Series([50.0] * len(df), index=df.index)
    
    closes = df['close']
    deltas = closes.diff()
    
    gains = deltas.where(deltas > 0, 0)
    losses = -deltas.where(deltas < 0, 0)
    
    avg_gains = gains.rolling(window=period, min_periods=1).mean()
    avg_losses = losses.rolling(window=period, min_periods=1).mean()
    
    # Replace zero losses with small value to avoid division by zero
    avg_losses = avg_losses.replace(0, 1e-10)
    
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    # Fill NaN values with 50 (neutral RSI)
    rsi = rsi.fillna(50.0)
    
    return rsi


def calculate_macd_series(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """
    Calculate MACD for all bars in DataFrame.
    
    Args:
        df: DataFrame with 'close' column
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
    
    Returns:
        Dictionary with 'macd', 'signal', and 'histogram' Series
    """
    if len(df) < slow + signal:
        zeros = pd.Series([0.0] * len(df), index=df.index)
        return {'macd': zeros, 'signal': zeros, 'histogram': zeros}
    
    closes = df['close']
    
    # Calculate EMAs
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    
    # MACD line
    macd_line = ema_fast - ema_slow
    
    # Signal line (EMA of MACD)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # Histogram
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def create_backtest_trade_chart(
    trade: pd.Series,
    bars_1min: pd.DataFrame,
    bars_15min: pd.DataFrame,
    fvgs: List[Dict],
    tpsl_levels: pd.DataFrame,
    stop_adjustments_list: List[Tuple[datetime, float]] = None,
    tp_adjustments_list: List[Tuple[datetime, float]] = None
) -> go.Figure:
    """Create an interactive Plotly chart for the trade (backtest format)."""
    import pytz
    ET = pytz.timezone('America/New_York')
    
    if stop_adjustments_list is None:
        stop_adjustments_list = []
    if tp_adjustments_list is None:
        tp_adjustments_list = []
    
    symbol = trade.get('symbol', 'UNKNOWN')
    if pd.isna(symbol):
        symbol = 'UNKNOWN'
    direction = trade.get('direction', 'unknown')
    if pd.isna(direction):
        direction = 'unknown'
    pattern_type = trade.get('pattern_type', 'FVG')
    if pd.isna(pattern_type):
        pattern_type = 'FVG'
    entry_time = trade.get('entry_datetime')
    exit_time = trade.get('exit_datetime')
    entry_price = trade.get('entry_price', 0)
    if pd.isna(entry_price):
        entry_price = trade.get('entry', 0)
        if pd.isna(entry_price):
            entry_price = 0
    exit_price = trade.get('exit_price', 0)
    if pd.isna(exit_price):
        exit_price = 0
    exit_reason = trade.get('exit_reason', 'Unknown')
    if pd.isna(exit_reason):
        exit_reason = 'Unknown'
    pnl_dollars = trade.get('pnl_dollars', 0)
    if pd.isna(pnl_dollars):
        pnl_dollars = 0
    
    # Pattern candle timestamps
    c1_time = trade.get('c1_datetime')
    c2_time = trade.get('c2_datetime')
    c3_time = trade.get('c3_datetime')
    
    # Sweep candle timestamp (for Reversal Sweep and Liquidity Sweep patterns)
    sweep_candle_time = trade.get('sweep_candle_datetime')
    
    # Reversal-specific timestamps (for Reversal Sweep patterns)
    is_reversal = trade.get('is_reversal', False)
    invalidation_candle_time = trade.get('invalidation_candle_datetime')
    confirmation_candle_time = trade.get('confirmation_datetime')
    
    # Get TP1 and TP2 levels
    tp1_price = trade.get('tp1_price')
    tp2_price = trade.get('tp2_price')
    
    # Get initial and final levels
    initial_stop = trade.get('initial_stop_loss', trade.get('original_stop_loss_post_readjust'))
    initial_tp = trade.get('initial_take_profit', trade.get('original_take_profit_post_readjust'))
    final_stop = trade.get('final_stop_loss', trade.get('stop_loss', initial_stop))
    final_tp = trade.get('final_take_profit', trade.get('take_profit', initial_tp))
    
    # Use stop_loss/take_profit if initial values are not available
    if pd.isna(initial_stop):
        initial_stop = trade.get('stop_loss')
    if pd.isna(initial_tp):
        initial_tp = trade.get('take_profit')
    if pd.isna(final_stop):
        final_stop = initial_stop
    if pd.isna(final_tp):
        final_tp = initial_tp
    
    # =========================================================================
    # DEFENSIVE VALIDATION: Detect inverted TP/SL levels
    # =========================================================================
    # For LONG trades: TP should be ABOVE entry, SL should be BELOW entry
    # For SHORT trades: TP should be BELOW entry, SL should be ABOVE entry
    direction = trade.get('direction', '')
    pattern_type = trade.get('pattern_type', '')
    
    def _check_level_inversion(level_name: str, level_value, entry: float, is_tp: bool) -> bool:
        """Returns True if level is inverted (wrong side of entry for trade direction)."""
        if level_value is None or pd.isna(level_value) or entry is None or pd.isna(entry):
            return False
        if direction == 'long':
            # Long: TP should be > entry, SL should be < entry
            if is_tp:
                return level_value < entry  # TP below entry is inverted for long
            else:
                return level_value > entry  # SL above entry is inverted for long
        elif direction == 'short':
            # Short: TP should be < entry, SL should be > entry
            if is_tp:
                return level_value > entry  # TP above entry is inverted for short
            else:
                return level_value < entry  # SL below entry is inverted for short
        return False
    
    # Check for inversions and warn
    inversions_detected = []
    if _check_level_inversion('initial_stop', initial_stop, entry_price, is_tp=False):
        inversions_detected.append(f"initial_stop={initial_stop:.2f}")
    if _check_level_inversion('initial_tp', initial_tp, entry_price, is_tp=True):
        inversions_detected.append(f"initial_tp={initial_tp:.2f}")
    if _check_level_inversion('final_stop', final_stop, entry_price, is_tp=False):
        inversions_detected.append(f"final_stop={final_stop:.2f}")
    if _check_level_inversion('final_tp', final_tp, entry_price, is_tp=True):
        inversions_detected.append(f"final_tp={final_tp:.2f}")
    if tp1_price is not None and _check_level_inversion('tp1', tp1_price, entry_price, is_tp=True):
        inversions_detected.append(f"tp1={tp1_price:.2f}")
    if tp2_price is not None and _check_level_inversion('tp2', tp2_price, entry_price, is_tp=True):
        inversions_detected.append(f"tp2={tp2_price:.2f}")
    
    if inversions_detected:
        print(f"  [VIZ WARNING] {symbol} {pattern_type}: INVERTED TP/SL detected for {direction.upper()} trade!")
        print(f"    Entry: {entry_price:.2f}, Inverted levels: {', '.join(inversions_detected)}")
        print(f"    This indicates a bug in trade creation logic - TP/SL on wrong side of entry.")
    
    # Aggregate to 5-minute bars for indicators
    bars_5min = aggregate_to_5min(bars_1min)
    
    # Calculate RSI and MACD indicators on 5-minute timeframe
    rsi_5min = calculate_rsi_series(bars_5min, period=14)
    macd_5min = calculate_macd_series(bars_5min, fast=12, slow=26, signal=9)
    
    # Calculate RSI and MACD indicators on 15-minute timeframe
    rsi_15min = calculate_rsi_series(bars_15min, period=14)
    macd_15min = calculate_macd_series(bars_15min, fast=12, slow=26, signal=9)
    
    # Create figure with subplots: main chart, 5min RSI, 5min MACD, 15min RSI, 15min MACD, DEBUG RSI
    # Note: shared_xaxes=False to prevent candlestick duplication bug in Plotly
    # Added 6th row for debugging - duplicate 5min RSI
    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=False,  # Disabled to fix candlestick duplication
        vertical_spacing=0.02,
        row_heights=[0.40, 0.12, 0.12, 0.12, 0.12, 0.12],  # Main chart 40%, each indicator 12%
        subplot_titles=('', 'RSI (5min)', 'MACD (5min)', 'RSI (15min)', 'MACD (15min)', 'RSI (5min) DEBUG'),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}], 
               [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # =========================================================================
    # LAYER 1 (BACKGROUND): FVG ZONES
    # =========================================================================
    for i, fvg in enumerate(fvgs):
        fvg_color = 'rgba(76, 175, 80, 0.15)' if fvg['type'] == 'long' else 'rgba(244, 67, 54, 0.15)'
        border_color = 'rgba(76, 175, 80, 0.5)' if fvg['type'] == 'long' else 'rgba(244, 67, 54, 0.5)'
        
        fig.add_shape(
            type="rect",
            x0=fvg['timestamp'],
            x1=fvg['entry_window_end'],
            y0=fvg['gap_low'],
            y1=fvg['gap_high'],
            fillcolor=fvg_color,
            line=dict(color=border_color, width=1),
            layer="below",
            name=f"FVG {fvg['type']}",
            xref='x',
            yref='y'
        )
        
        # Add FVG label (invisible scatter for hover)
        fig.add_trace(go.Scatter(
            x=[fvg['timestamp'] + timedelta(minutes=7)],
            y=[(fvg['gap_low'] + fvg['gap_high']) / 2],
            mode='markers',
            marker=dict(size=1, opacity=0),
            showlegend=False,
            hovertemplate=(
                f"<b>FVG ({fvg['type'].upper()})</b><br>"
                f"Gap: ${fvg['gap_low']:.2f} - ${fvg['gap_high']:.2f}<br>"
                f"Size: ${fvg['gap_size']:.2f}<br>"
                f"Valid until: {fvg['entry_window_end'].strftime('%H:%M')}<br>"
                f"<extra></extra>"
            )
        ), row=1, col=1)
    
    # =========================================================================
    # LAYER 2: 15-MINUTE CANDLES (TRANSPARENT BACKGROUND)
    # Highlight c1, c2, c3 candles with special colors
    # For Reversal Sweep and Liquidity Sweep, also highlight sweep candle
    # =========================================================================
    for _, bar15 in bars_15min.iterrows():
        bar_start = bar15['timestamp']
        bar_end = bar_start + timedelta(minutes=15)
        
        # Check if this is a pattern candle (c1, c2, c3, sweep_candle, or reversal-specific candles)
        is_c1 = c1_time is not None and not pd.isna(c1_time) and abs((bar_start - c1_time).total_seconds()) < 60
        is_c2 = c2_time is not None and not pd.isna(c2_time) and abs((bar_start - c2_time).total_seconds()) < 60
        is_c3 = c3_time is not None and not pd.isna(c3_time) and abs((bar_start - c3_time).total_seconds()) < 60
        is_sweep = sweep_candle_time is not None and not pd.isna(sweep_candle_time) and abs((bar_start - sweep_candle_time).total_seconds()) < 60
        
        # Reversal-specific candles
        is_invalidation = is_reversal and invalidation_candle_time is not None and not pd.isna(invalidation_candle_time) and abs((bar_start - invalidation_candle_time).total_seconds()) < 60
        is_confirmation = is_reversal and confirmation_candle_time is not None and not pd.isna(confirmation_candle_time) and abs((bar_start - confirmation_candle_time).total_seconds()) < 60
        
        is_bullish = bar15['close'] >= bar15['open']
        
        # Different colors for pattern candles
        # Priority order: Confirmation > Invalidation > Sweep > C3 > C2 > C1
        if is_confirmation:
            fill_color = 'rgba(0, 200, 83, 0.6)'  # Bright Green for Confirmation candle
            line_color = 'rgba(0, 200, 83, 1.0)'
            candle_label = 'CONFIRM'
        elif is_invalidation:
            fill_color = 'rgba(255, 23, 68, 0.6)'  # Bright Red for Invalidation candle
            line_color = 'rgba(255, 23, 68, 1.0)'
            candle_label = 'INVALID'
        elif is_c1:
            fill_color = 'rgba(255, 193, 7, 0.5)'  # Gold/Amber for C1
            line_color = 'rgba(255, 193, 7, 0.9)'
            candle_label = 'C1'
        elif is_c2:
            fill_color = 'rgba(156, 39, 176, 0.5)'  # Purple for C2
            line_color = 'rgba(156, 39, 176, 0.9)'
            candle_label = 'C2'
        elif is_c3:
            fill_color = 'rgba(0, 188, 212, 0.5)'  # Cyan for C3
            line_color = 'rgba(0, 188, 212, 0.9)'
            candle_label = 'C3'
        elif is_sweep:
            fill_color = 'rgba(255, 87, 34, 0.5)'  # Deep Orange for Sweep candle
            line_color = 'rgba(255, 87, 34, 0.9)'
            candle_label = 'SWEEP'
        elif is_bullish:
            fill_color = 'rgba(38, 166, 154, 0.35)'
            line_color = 'rgba(38, 166, 154, 0.6)'
            candle_label = None
        else:
            fill_color = 'rgba(239, 83, 80, 0.35)'
            line_color = 'rgba(239, 83, 80, 0.6)'
            candle_label = None
        
        # Draw candle body (rectangle from open to close)
        body_low = min(bar15['open'], bar15['close'])
        body_high = max(bar15['open'], bar15['close'])
        
        fig.add_shape(
            type="rect",
            x0=bar_start,
            x1=bar_end,
            y0=body_low,
            y1=body_high,
            fillcolor=fill_color,
            line=dict(color=line_color, width=2 if candle_label else 1),
            layer="below"
        )
        
        # Draw wicks (vertical lines for high/low)
        wick_x = bar_start + timedelta(minutes=7.5)  # Center of the bar
        
        # Upper wick
        if bar15['high'] > body_high:
            fig.add_shape(
                type="line",
                x0=wick_x,
                x1=wick_x,
                y0=body_high,
                y1=bar15['high'],
                line=dict(color=line_color, width=2 if candle_label else 1),
                layer="below"
            )
        
        # Lower wick
        if bar15['low'] < body_low:
            fig.add_shape(
                type="line",
                x0=wick_x,
                x1=wick_x,
                y0=bar15['low'],
                y1=body_low,
                line=dict(color=line_color, width=2 if candle_label else 1),
                layer="below"
            )
        
        # Add label for pattern candles
        if candle_label:
            fig.add_annotation(
                x=bar_start + timedelta(minutes=7.5),
                y=bar15['high'] + (bar15['high'] - bar15['low']) * 0.1,
                text=f"<b>{candle_label}</b>",
                showarrow=False,
                font=dict(size=12, color=line_color),
                bgcolor='white',
                bordercolor=line_color,
                borderwidth=1
            )
    
    # Add invisible scatter for 15-min hover info - only on main chart
    if not bars_15min.empty:
        scatter_15min = go.Scatter(
            x=bars_15min['timestamp'] + pd.Timedelta(minutes=7.5),
            y=(bars_15min['high'] + bars_15min['low']) / 2,
            mode='markers',
            marker=dict(size=1, opacity=0),
            name='15min',
            showlegend=True,
            hovertemplate=[
                f"<b>15min Bar</b><br>Time: {t.strftime('%H:%M')}<br>"
                f"O: ${o:.2f}<br>H: ${h:.2f}<br>L: ${l:.2f}<br>C: ${c:.2f}<extra></extra>"
                for t, o, h, l, c in zip(
                    bars_15min['timestamp'], bars_15min['open'], bars_15min['high'],
                    bars_15min['low'], bars_15min['close']
                )
            ],
            xaxis='x',
            yaxis='y'
        )
        fig.add_trace(scatter_15min, row=1, col=1)
    
    # =========================================================================
    # LAYER 3: 1-MINUTE CANDLES (MAIN LAYER)
    # =========================================================================
    # NOTE: Candlestick will be added AFTER all indicator subplots to prevent Plotly bug
    # Store candlestick data for later addition
    candlestick_data = {
        'x': bars_1min['timestamp'],
        'open': bars_1min['open'],
        'high': bars_1min['high'],
        'low': bars_1min['low'],
        'close': bars_1min['close'],
        'volume': bars_1min['volume']
    }
    
    # =========================================================================
    # LAYER 4: ENTRY AND EXIT MARKERS
    # =========================================================================
    
    # Check if this is an unfilled trade - check both 'filled' column and 'result' field
    # Note: entry_time/exit_time may have been set to pattern timestamps for visualization purposes
    filled_status = str(trade.get('filled', '')).upper()
    trade_result = trade.get('result', '').upper()
    
    # Check if trade is unfilled based on:
    # 1. 'filled' column == 'NO'
    # 2. 'result' field indicates unfilled
    # 3. '_viz_unfilled_trade' flag set during preprocessing (when entry/exit times were NaN)
    is_unfilled_by_filled = filled_status == 'NO'
    is_unfilled_by_result = trade_result in (
        'NOT FILLED', 'UNFILLED', 'EXPIRED', 
        'REVERSAL_CONFIRMATION_EXPIRED', 'REVERSAL_DEEP_RETRACE',
        'PATTERN EXPIRED BEFORE ENTRY'
    )
    is_unfilled_by_viz_flag = trade.get('_viz_unfilled_trade', False)
    is_unfilled_trade = is_unfilled_by_filled or is_unfilled_by_result or is_unfilled_by_viz_flag
    
    # Entry and exit markers - only show actual entry/exit for filled trades
    # For unfilled trades, show grayed-out planned entry marker
    entry_color = '#2196F3' if direction == 'long' else '#FF9800'
    entry_symbol = 'triangle-up' if direction == 'long' else 'triangle-down'
    
    if is_unfilled_trade:
        # For unfilled trades, show planned entry as a grayed-out marker with open symbol
        planned_entry = trade.get('entry', trade.get('entry_price', entry_price))
        if planned_entry and not pd.isna(planned_entry):
            # Use pattern timestamp for x position
            pattern_ts = c1_time or c2_time or c3_time or sweep_candle_time
            if pattern_ts and not pd.isna(pattern_ts):
                # Use open symbol variant to indicate unfilled
                unfilled_symbol = 'triangle-up-open' if direction == 'long' else 'triangle-down-open'
                # Get filled_reason for hover info
                filled_reason = trade.get('filled_reason', trade_result if trade_result else 'Not filled')
                if pd.isna(filled_reason) or not filled_reason:
                    filled_reason = 'Not filled'
                planned_entry_trace = go.Scatter(
                    x=[pattern_ts],
                    y=[planned_entry],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='rgba(150, 150, 150, 0.8)',  # Grayed out
                        symbol=unfilled_symbol,
                        line=dict(width=2, color='gray')
                    ),
                    name='Planned Entry (unfilled)',
                    hovertemplate=(
                        f"<b>PLANNED ENTRY ({direction.upper()}) - UNFILLED</b><br>"
                        f"Planned Price: ${planned_entry:.2f}<br>"
                        f"Reason: {filled_reason}<br>"
                        f"<extra></extra>"
                    ),
                    xaxis='x',
                    yaxis='y'
                )
                fig.add_trace(planned_entry_trace, row=1, col=1)
        # No exit marker for unfilled trades
    else:
        # For filled trades, show actual entry
        entry_trace = go.Scatter(
            x=[entry_time],
            y=[entry_price],
            mode='markers',
            marker=dict(
                size=15,
                color=entry_color,
                symbol=entry_symbol,
                line=dict(width=2, color='black')
            ),
            name='Entry',
            hovertemplate=(
                f"<b>ENTRY ({direction.upper()})</b><br>"
                f"Time: {entry_time}<br>"
                f"Price: ${entry_price:.2f}<br>"
                f"<extra></extra>"
            ),
            xaxis='x',
            yaxis='y'
        )
        fig.add_trace(entry_trace, row=1, col=1)
        
        # Exit marker - only for filled trades
        exit_color = '#4CAF50' if pnl_dollars > 0 else '#F44336'
        
        exit_trace = go.Scatter(
            x=[exit_time],
            y=[exit_price],
            mode='markers',
            marker=dict(
                size=15,
                color=exit_color,
                symbol='x',
                line=dict(width=2, color='black')
            ),
            name='Exit',
            hovertemplate=(
                f"<b>EXIT ({exit_reason})</b><br>"
                f"Time: {exit_time}<br>"
                f"Price: ${exit_price:.2f}<br>"
                f"P&L: ${pnl_dollars:.2f}<br>"
                f"<extra></extra>"
            ),
            xaxis='x',
            yaxis='y'
        )
        fig.add_trace(exit_trace, row=1, col=1)
    
    # =========================================================================
    # LAYER 5: TP/SL LEVELS DURING TRADE
    # =========================================================================
    if not tpsl_levels.empty:
        # Stop loss line - explicitly set to row 1 only
        sl_trace = go.Scatter(
            x=tpsl_levels['timestamp'],
            y=tpsl_levels['stop_level'],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Stop Loss',
            hovertemplate="SL: $%{y:.2f}<extra></extra>",
            xaxis='x',
            yaxis='y'
        )
        fig.add_trace(sl_trace, row=1, col=1)
        
        # Take profit line - explicitly set to row 1 only
        tp_trace = go.Scatter(
            x=tpsl_levels['timestamp'],
            y=tpsl_levels['tp_level'],
            mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name='Take Profit',
            hovertemplate="TP: $%{y:.2f}<extra></extra>",
            xaxis='x',
            yaxis='y'
        )
        fig.add_trace(tp_trace, row=1, col=1)
    
    # =========================================================================
    # REFERENCE LINES AND ADJUSTMENT LEVELS
    # =========================================================================
    
    num_stop_adj = trade.get('num_stop_adjustments', 0)
    num_tp_adj = trade.get('num_tp_adjustments', 0)
    
    # Collect all price levels for y-axis range calculation
    all_price_levels = [entry_price, exit_price]
    if not pd.isna(initial_stop):
        all_price_levels.append(initial_stop)
    if not pd.isna(initial_tp):
        all_price_levels.append(initial_tp)
    if not pd.isna(final_stop):
        all_price_levels.append(final_stop)
    if not pd.isna(final_tp):
        all_price_levels.append(final_tp)
    if tp1_price is not None and not pd.isna(tp1_price):
        all_price_levels.append(tp1_price)
    if tp2_price is not None and not pd.isna(tp2_price):
        all_price_levels.append(tp2_price)
    
    # Add adjustment values to price levels
    for _, val in stop_adjustments_list:
        all_price_levels.append(val)
    for _, val in tp_adjustments_list:
        all_price_levels.append(val)
    
    # Add bar range
    if not bars_1min.empty:
        all_price_levels.append(bars_1min['high'].max())
        all_price_levels.append(bars_1min['low'].min())
    
    # Entry price line
    fig.add_hline(
        y=entry_price,
        line_dash="solid",
        line_color="rgba(33, 150, 243, 0.7)",
        annotation_text=f"  Entry: ${entry_price:.2f}  ",
        annotation_position="left",
        annotation=dict(font=dict(size=11), bgcolor="rgba(255,255,255,0.8)")
    )
    
    # Initial stop line (dotted, faded)
    if not pd.isna(initial_stop):
        fig.add_hline(
            y=initial_stop,
            line_dash="dot",
            line_color="rgba(255, 0, 0, 0.4)",
            annotation_text=f"  Init SL: ${initial_stop:.2f}  ",
            annotation_position="right",
            annotation=dict(font=dict(size=10), bgcolor="rgba(255,255,255,0.8)")
        )
    
    # TP1 line (partial exit target)
    if tp1_price is not None and not pd.isna(tp1_price):
        fig.add_hline(
            y=tp1_price,
            line_dash="dot",
            line_color="rgba(0, 180, 0, 0.4)",
            annotation_text=f"  TP1: ${tp1_price:.2f}  ",
            annotation_position="right",
            annotation=dict(font=dict(size=10), bgcolor="rgba(255,255,255,0.8)")
        )
    elif not pd.isna(initial_tp):
        # Fall back to initial_tp if tp1_price not available
        fig.add_hline(
            y=initial_tp,
            line_dash="dot",
            line_color="rgba(0, 180, 0, 0.4)",
            annotation_text=f"  Init TP: ${initial_tp:.2f}  ",
            annotation_position="right",
            annotation=dict(font=dict(size=10), bgcolor="rgba(255,255,255,0.8)")
        )
    
    # TP2 line (runner target) - if different from TP1
    if tp2_price is not None and not pd.isna(tp2_price):
        if tp1_price is None or pd.isna(tp1_price) or abs(tp2_price - tp1_price) > 0.01:
            fig.add_hline(
                y=tp2_price,
                line_dash="dot",
                line_color="rgba(0, 140, 0, 0.4)",
                annotation_text=f"  TP2: ${tp2_price:.2f}  ",
                annotation_position="right",
                annotation=dict(font=dict(size=10), bgcolor="rgba(255,255,255,0.8)")
            )
    
    # Final stop line (if different from initial)
    if not pd.isna(final_stop) and not pd.isna(initial_stop) and abs(final_stop - initial_stop) > 0.01:
        fig.add_hline(
            y=final_stop,
            line_dash="dashdot",
            line_color="rgba(255, 100, 0, 0.6)",
            annotation_text=f"  Final SL: ${final_stop:.2f} ({num_stop_adj} adj)  ",
            annotation_position="right",
            annotation=dict(font=dict(size=10), bgcolor="rgba(255,255,255,0.8)")
        )
    
    # Final TP line (if different from initial)
    if not pd.isna(final_tp) and not pd.isna(initial_tp) and abs(final_tp - initial_tp) > 0.01:
        fig.add_hline(
            y=final_tp,
            line_dash="dashdot",
            line_color="rgba(0, 200, 100, 0.6)",
            annotation_text=f"  Final TP: ${final_tp:.2f} ({num_tp_adj} ext)  ",
            annotation_position="right",
            annotation=dict(font=dict(size=10), bgcolor="rgba(255,255,255,0.8)")
        )
    
    # =========================================================================
    # LAYER 6: STOP ADJUSTMENT LEVELS (horizontal lines at each adjustment)
    # =========================================================================
    for i, (adj_time, adj_val) in enumerate(stop_adjustments_list):
        # Draw a short horizontal line segment at each stop adjustment level
        # extending from the adjustment time to exit or next adjustment
        end_time = exit_time
        if i + 1 < len(stop_adjustments_list):
            end_time = stop_adjustments_list[i + 1][0]
        
        fig.add_shape(
            type="line",
            x0=adj_time,
            x1=end_time,
            y0=adj_val,
            y1=adj_val,
            line=dict(color="rgba(255, 100, 0, 0.5)", width=1.5, dash="dot"),
            layer="above"
        )
        
        # Add marker at adjustment point
        fig.add_trace(go.Scatter(
            x=[adj_time],
            y=[adj_val],
            mode='markers',
            marker=dict(size=6, color='rgba(255, 100, 0, 0.8)', symbol='diamond'),
            showlegend=False,
            hovertemplate=f"<b>SL Adj #{i+1}</b><br>Time: {adj_time}<br>New SL: ${adj_val:.2f}<extra></extra>"
        ))
    
    # =========================================================================
    # LAYER 7: TP ADJUSTMENT LEVELS (horizontal lines at each adjustment)
    # =========================================================================
    for i, (adj_time, adj_val) in enumerate(tp_adjustments_list):
        # Draw a short horizontal line segment at each TP adjustment level
        end_time = exit_time
        if i + 1 < len(tp_adjustments_list):
            end_time = tp_adjustments_list[i + 1][0]
        
        fig.add_shape(
            type="line",
            x0=adj_time,
            x1=end_time,
            y0=adj_val,
            y1=adj_val,
            line=dict(color="rgba(0, 200, 100, 0.5)", width=1.5, dash="dot"),
            layer="above"
        )
        
        # Add marker at adjustment point
        fig.add_trace(go.Scatter(
            x=[adj_time],
            y=[adj_val],
            mode='markers',
            marker=dict(size=6, color='rgba(0, 200, 100, 0.8)', symbol='diamond'),
            showlegend=False,
            hovertemplate=f"<b>TP Ext #{i+1}</b><br>Time: {adj_time}<br>New TP: ${adj_val:.2f}<extra></extra>"
        ))
    
    # Shade trade period - with extension visualization
    if entry_time is not None and exit_time is not None:
        has_adjustments = num_stop_adj > 0 or num_tp_adj > 0
        
        if CONFIG_AVAILABLE and has_adjustments:
            # Get max duration for this trade's direction
            trading_tf = trade.get('trading_timeframe', '15min')
            max_duration = get_trade_duration_minutes(direction, trading_tf)
            original_window_end = entry_time + timedelta(minutes=max_duration)
            
            # If trade extended beyond original window, show both regions
            if exit_time > original_window_end:
                # Initial window (light gray)
                fig.add_vrect(
                    x0=entry_time,
                    x1=original_window_end,
                    fillcolor="rgba(100, 100, 100, 0.08)",
                    line_width=0,
                    annotation_text="Initial Window",
                    annotation_position="top left"
                )
                # Extended window (light blue)
                fig.add_vrect(
                    x0=original_window_end,
                    x1=exit_time,
                    fillcolor="rgba(100, 150, 255, 0.12)",
                    line_width=0,
                    annotation_text="Extended",
                    annotation_position="top left"
                )
            else:
                # Trade ended within original window
                fig.add_vrect(
                    x0=entry_time,
                    x1=exit_time,
                    fillcolor="rgba(100, 100, 100, 0.08)",
                    line_width=0,
                    annotation_text="Trade Period",
                    annotation_position="top left"
                )
        else:
            # No adjustments or config not available - single gray region
            fig.add_vrect(
                x0=entry_time,
                x1=exit_time,
                fillcolor="rgba(100, 100, 100, 0.08)",
                line_width=0,
                annotation_text="Trade Period",
                annotation_position="top left"
            )
    
    # =========================================================================
    # REVERSAL CONFIRMATION WINDOW SHADING (light purple)
    # =========================================================================
    if is_reversal:
        # Get confirmation window parameters - prefer datetime version if available
        confirmation_start = (
            trade.get('confirmation_start_datetime') or 
            trade.get('confirmation_start_time') or 
            trade.get('invalidating_candle_timestamp') or
            invalidation_candle_time
        )
        confirmation_window_minutes = trade.get('confirmation_window_minutes', 45)
        if confirmation_window_minutes == '' or pd.isna(confirmation_window_minutes):
            confirmation_window_minutes = 45
        
        if confirmation_start:
            try:
                conf_start_ts = pd.to_datetime(confirmation_start)
                if conf_start_ts.tzinfo is None:
                    conf_start_ts = conf_start_ts.tz_localize(ET)
                else:
                    conf_start_ts = conf_start_ts.tz_convert(ET)
                
                conf_end_ts = conf_start_ts + timedelta(minutes=confirmation_window_minutes)
                
                # Shade the confirmation window in light purple (only on main chart)
                fig.add_vrect(
                    x0=conf_start_ts,
                    x1=conf_end_ts,
                    fillcolor="rgba(156, 39, 176, 0.08)",  # Light purple
                    line_width=1,
                    line_color="rgba(156, 39, 176, 0.3)",
                    annotation_text="Confirmation Window",
                    annotation_position="bottom left",
                    annotation=dict(font=dict(size=10, color="rgba(156, 39, 176, 0.8)")),
                    row=1, col=1
                )
            except Exception:
                pass  # Silently skip if timestamp parsing fails
    
    # =========================================================================
    # RSI SUBPLOT (5min) - ROW 2
    # =========================================================================
    # IMPORTANT: Add ALL indicator subplots BEFORE adding the candlestick
    # This prevents Plotly bug where candlestick appears on other subplots
    
    # Add RSI line for 5min
    fig.add_trace(go.Scatter(
        x=bars_5min['timestamp'],
        y=rsi_5min,
        mode='lines',
        name='RSI (5min)',
        line=dict(color='purple', width=2),
        hovertemplate='RSI (5min): %{y:.2f}<extra></extra>',
        xaxis='x2',
        yaxis='y2'
    ), row=2, col=1)
    
    # Add RSI zones (30/70)
    fig.add_hrect(
        y0=70, y1=100,
        fillcolor="rgba(255, 0, 0, 0.1)",
        layer="below",
        line_width=0,
        row=2, col=1
    )
    fig.add_hrect(
        y0=0, y1=30,
        fillcolor="rgba(0, 255, 0, 0.1)",
        layer="below",
        line_width=0,
        row=2, col=1
    )
    
    # Add RSI reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)
    
    # =========================================================================
    # MACD SUBPLOT (5min) - ROW 3
    # =========================================================================
    # Add MACD line
    fig.add_trace(go.Scatter(
        x=bars_5min['timestamp'],
        y=macd_5min['macd'],
        mode='lines',
        name='MACD (5min)',
        line=dict(color='blue', width=2),
        hovertemplate='MACD (5min): %{y:.4f}<extra></extra>'
    ), row=3, col=1)
    
    # Add Signal line
    fig.add_trace(go.Scatter(
        x=bars_5min['timestamp'],
        y=macd_5min['signal'],
        mode='lines',
        name='Signal (5min)',
        line=dict(color='orange', width=2),
        hovertemplate='Signal (5min): %{y:.4f}<extra></extra>'
    ), row=3, col=1)
    
    # Add Histogram (bar chart)
    colors_5min = ['green' if h >= 0 else 'red' for h in macd_5min['histogram']]
    fig.add_trace(go.Bar(
        x=bars_5min['timestamp'],
        y=macd_5min['histogram'],
        name='Histogram (5min)',
        marker_color=colors_5min,
        opacity=0.6,
        hovertemplate='Histogram (5min): %{y:.4f}<extra></extra>'
    ), row=3, col=1)
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1)
    
    # =========================================================================
    # RSI SUBPLOT (15min) - ROW 4
    # =========================================================================
    # Add RSI line
    fig.add_trace(go.Scatter(
        x=bars_15min['timestamp'],
        y=rsi_15min,
        mode='lines',
        name='RSI (15min)',
        line=dict(color='purple', width=2),
        hovertemplate='RSI (15min): %{y:.2f}<extra></extra>'
    ), row=4, col=1)
    
    # Add RSI zones (30/70)
    fig.add_hrect(
        y0=70, y1=100,
        fillcolor="rgba(255, 0, 0, 0.1)",
        layer="below",
        line_width=0,
        row=4, col=1
    )
    fig.add_hrect(
        y0=0, y1=30,
        fillcolor="rgba(0, 255, 0, 0.1)",
        layer="below",
        line_width=0,
        row=4, col=1
    )
    
    # Add RSI reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=4, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=4, col=1)
    
    # =========================================================================
    # MACD SUBPLOT (15min) - ROW 5
    # =========================================================================
    # Add MACD line
    fig.add_trace(go.Scatter(
        x=bars_15min['timestamp'],
        y=macd_15min['macd'],
        mode='lines',
        name='MACD (15min)',
        line=dict(color='blue', width=2),
        hovertemplate='MACD (15min): %{y:.4f}<extra></extra>'
    ), row=5, col=1)
    
    # Add Signal line
    fig.add_trace(go.Scatter(
        x=bars_15min['timestamp'],
        y=macd_15min['signal'],
        mode='lines',
        name='Signal (15min)',
        line=dict(color='orange', width=2),
        hovertemplate='Signal (15min): %{y:.4f}<extra></extra>'
    ), row=5, col=1)
    
    # Add Histogram (bar chart)
    colors_15min = ['green' if h >= 0 else 'red' for h in macd_15min['histogram']]
    fig.add_trace(go.Bar(
        x=bars_15min['timestamp'],
        y=macd_15min['histogram'],
        name='Histogram (15min)',
        marker_color=colors_15min,
        opacity=0.6,
        hovertemplate='Histogram (15min): %{y:.4f}<extra></extra>'
    ), row=5, col=1)
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=5, col=1)
    
    # =========================================================================
    # DEBUG RSI SUBPLOT (5min) - ROW 6
    # =========================================================================
    # Add duplicate 5min RSI to row 6 to see if it renders correctly
    fig.add_trace(go.Scatter(
        x=bars_5min['timestamp'],
        y=rsi_5min,
        mode='lines',
        name='RSI (5min) DEBUG',
        line=dict(color='orange', width=2),
        hovertemplate='RSI (5min) DEBUG: %{y:.2f}<extra></extra>',
        xaxis='x6',
        yaxis='y6'
    ), row=6, col=1)
    
    # Add RSI zones (30/70) for debug chart
    fig.add_hrect(
        y0=70, y1=100,
        fillcolor="rgba(255, 0, 0, 0.1)",
        layer="below",
        line_width=0,
        row=6, col=1
    )
    fig.add_hrect(
        y0=0, y1=30,
        fillcolor="rgba(0, 255, 0, 0.1)",
        layer="below",
        line_width=0,
        row=6, col=1
    )
    
    # Add RSI reference lines for debug chart
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=6, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=6, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=6, col=1)
    
    # =========================================================================
    # NOW ADD CANDLESTICK TO ROW 1 (AFTER ALL INDICATOR SUBPLOTS)
    # =========================================================================
    candlestick_trace = go.Candlestick(
        x=candlestick_data['x'],
        open=candlestick_data['open'],
        high=candlestick_data['high'],
        low=candlestick_data['low'],
        close=candlestick_data['close'],
        name='1min',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        hoverinfo='text',
        hovertext=[
            f"Time: {t}<br>O: ${o:.2f}<br>H: ${h:.2f}<br>L: ${l:.2f}<br>C: ${c:.2f}<br>Vol: {v:,.0f}"
            for t, o, h, l, c, v in zip(
                candlestick_data['x'], candlestick_data['open'], candlestick_data['high'], 
                candlestick_data['low'], candlestick_data['close'], candlestick_data['volume']
            )
        ]
    )
    # Add to row 1 only
    fig.add_trace(candlestick_trace, row=1, col=1)
    # Move it to the front (index 0) so it renders first (behind other traces)
    fig.data = (fig.data[-1],) + fig.data[:-1]
    
    # =========================================================================
    # LAYOUT
    # =========================================================================
    pnl_sign = '+' if pnl_dollars > 0 else ''
    long_fvgs = sum(1 for f in fvgs if f['type'] == 'long')
    short_fvgs = sum(1 for f in fvgs if f['type'] == 'short')
    
    win_loss = trade.get('win_loss', 'N/A')
    if pd.isna(win_loss):
        win_loss = 'N/A'
    backtest_date = trade.get('backtest_date', '')
    if pd.isna(backtest_date):
        backtest_date = ''
    
    # Format c1_time for title - use sweep_candle_time or entry_time as fallback
    c1_time_str = ""
    pattern_time = None
    for ts_candidate in [c1_time, sweep_candle_time, entry_time]:
        if ts_candidate is not None and not pd.isna(ts_candidate):
            pattern_time = ts_candidate
            break
    
    if pattern_time is not None:
        if hasattr(pattern_time, 'strftime'):
            c1_time_str = pattern_time.strftime('%Y-%m-%d %H:%M')
        else:
            c1_time_str = str(pattern_time)
    else:
        c1_time_str = backtest_date if backtest_date else "Unknown Date"
    
    entry_time_str = ""
    if entry_time is not None and not pd.isna(entry_time):
        if hasattr(entry_time, 'strftime'):
            entry_time_str = entry_time.strftime('%H:%M')
        else:
            entry_time_str = str(entry_time)
    
    # Check if trade is unfilled - check 'filled' column, 'result' field, and _viz_unfilled_trade flag
    # Note: entry_time/exit_time may have been set to pattern timestamps for visualization purposes
    filled_status = str(trade.get('filled', '')).upper()
    trade_result_upper = trade.get('result', '').upper()
    is_unfilled_by_filled = filled_status == 'NO'
    is_unfilled_by_result = trade_result_upper in (
        'NOT FILLED', 'UNFILLED', 'EXPIRED', 
        'REVERSAL_CONFIRMATION_EXPIRED', 'REVERSAL_DEEP_RETRACE',
        'PATTERN EXPIRED BEFORE ENTRY'
    )
    is_unfilled_by_viz_flag = trade.get('_viz_unfilled_trade', False)
    is_unfilled = is_unfilled_by_filled or is_unfilled_by_result or is_unfilled_by_viz_flag
    unfilled_marker = " (UNFILLED)" if is_unfilled else ""
    
    # Title format: c1_timestamp | symbol | direction | pattern_type
    # Build main title line (always shown) - use simple format without HTML for reliability
    main_title = f"{c1_time_str} | {symbol} | {direction.upper()} | {pattern_type}{unfilled_marker}"
    
    # DEBUG: Print title components
    print(f"[DEBUG TITLE] c1_time_str='{c1_time_str}', symbol='{symbol}', direction='{direction}', pattern_type='{pattern_type}'")
    print(f"[DEBUG TITLE] main_title='{main_title}'")
    
    if is_unfilled:
        # Unfilled trade - show different info
        result_reason = trade.get('result', trade.get('filled_reason', 'Unknown'))
        if pd.isna(result_reason):
            result_reason = 'Unknown'
        title = f"{main_title}"
    else:
        title = f"{main_title}"
    
    # Calculate y-axis range with padding
    if all_price_levels:
        y_min = min(all_price_levels)
        y_max = max(all_price_levels)
        y_range = y_max - y_min
        y_padding = y_range * 0.15  # 15% padding on each side
        y_axis_range = [y_min - y_padding, y_max + y_padding]
    else:
        y_axis_range = None
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        template='plotly_white',
        height=1600,  # Taller chart to accommodate 6 subplots (added debug row)
        margin=dict(l=100, r=150, t=100, b=60),  # More padding for labels
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.9)",
            visible=False  # Hidden by default, shown on hover
        ),
        hovermode='x unified',
        annotations=[
            dict(
                text="🔑",
                xref="paper", yref="paper",
                x=0.01, y=0.99,
                xanchor="left", yanchor="top",
                showarrow=False,
                font=dict(size=16),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.3)",
                borderwidth=1,
                borderpad=4
            )
        ]
    )
    
    # Update axes for main chart (row 1)
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    if y_axis_range:
        fig.update_yaxes(range=y_axis_range, row=1, col=1)
    
    # Update axes for 5min RSI subplot (row 2)
    fig.update_xaxes(title_text="", row=2, col=1, showticklabels=True)
    fig.update_yaxes(title_text="RSI (5min)", row=2, col=1, range=[0, 100])
    
    # Update axes for 5min MACD subplot (row 3)
    fig.update_xaxes(title_text="", row=3, col=1, showticklabels=True)
    fig.update_yaxes(title_text="MACD (5min)", row=3, col=1)
    
    # Update axes for 15min RSI subplot (row 4)
    fig.update_xaxes(title_text="", row=4, col=1, showticklabels=True)
    fig.update_yaxes(title_text="RSI (15min)", row=4, col=1, range=[0, 100])
    
    # Update axes for 15min MACD subplot (row 5)
    fig.update_xaxes(title_text="", row=5, col=1, showticklabels=True)
    fig.update_yaxes(title_text="MACD (15min)", row=5, col=1)
    
    # Update axes for DEBUG RSI subplot (row 6)
    fig.update_xaxes(title_text="Time", row=6, col=1)
    fig.update_yaxes(title_text="RSI (5min) DEBUG", row=6, col=1, range=[0, 100])
    
    # Manually sync x-axes for all subplots (since shared_xaxes=False to fix candlestick issue)
    # This ensures zooming/panning works together across all subplots
    for row in range(2, 7):  # Updated to include row 6
        fig.update_xaxes(matches='x', row=row, col=1)
    
    # CRITICAL FIX: Explicitly ensure candlestick only appears on row 1
    # Plotly bug: candlestick charts can duplicate on other subplots even with row/col specified
    # We need to explicitly hide the candlestick from rows 2-5 by setting their visibility
    fig.update_layout(
        xaxis=dict(domain=[0, 1], anchor='y'),
        xaxis2=dict(domain=[0, 1], anchor='y2', matches='x'),
        xaxis3=dict(domain=[0, 1], anchor='y3', matches='x'),
        xaxis4=dict(domain=[0, 1], anchor='y4', matches='x'),
        xaxis5=dict(domain=[0, 1], anchor='y5', matches='x'),
        xaxis6=dict(domain=[0, 1], anchor='y6', matches='x'),
    )
    
    # Force all row-1-only traces to use xaxis='x' and yaxis='y'
    # This prevents them from appearing on other subplots
    for i, trace in enumerate(fig.data):
        # OHLC (replaced candlestick) and other row-1 traces
        if (isinstance(trace, (go.Candlestick, go.Ohlc)) or 
            (hasattr(trace, 'name') and trace.name in ['Entry', 'Exit', 'Planned Entry (unfilled)', 
                                                         'Stop Loss', 'Take Profit', '15min', '1min'])):
            trace.xaxis = 'x'
            trace.yaxis = 'y'
    
    # Additional fix: Ensure OHLC/candlestick is not visible on xaxis2-xaxis5
    # by updating the trace's visible property for those subplots
    fig.update_traces(
        selector=dict(type='ohlc'),
        xaxis='x',  # Force to use only x (row 1)
        yaxis='y'   # Force to use only y (row 1)
    )
    fig.update_traces(
        selector=dict(type='candlestick'),
        xaxis='x',  # Force to use only x (row 1)
        yaxis='y'   # Force to use only y (row 1)
    )
    
    return fig


# =============================================================================
# MAIN INTERACTIVE LOOP - OPTIMIZER
# =============================================================================

def select_optimizer_csv_file(csv_files: List[Path]) -> Optional[Path]:
    """Interactive file selection for optimizer files."""
    print("\n" + "="*60)
    print("AVAILABLE OPTIMIZER TRADE DETAIL FILES")
    print("="*60)
    
    for i, f in enumerate(csv_files, 1):
        # Count trades in file
        try:
            df = pd.read_csv(f)
            count = len(df)
        except:
            count = "?"
        print(f"  [{i}] {f.name} ({count} trades)")
    
    print(f"\n  [0] Back to main menu")
    print()
    
    while True:
        try:
            choice = input("Select a file number: ").strip()
            if choice == '0' or choice.lower() in ('b', 'back'):
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(csv_files):
                return csv_files[idx]
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a number.")


def run_optimizer_visualization():
    """Run optimizer trade visualization loop."""
    csv_files = list_optimizer_csv_files()
    if not csv_files:
        print("No optimizer trade detail CSV files found.")
        return
    
    while True:
        # Select file
        selected_file = select_optimizer_csv_file(csv_files)
        if selected_file is None:
            return
        
        print(f"\nLoading {selected_file.name}...")
        df = load_trade_csv(selected_file)
        print(f"Loaded {len(df)} trades")
        
        # Show sample trade IDs
        sample_ids = df['trade_id'].head(10).tolist()
        print(f"\nSample trade IDs: {', '.join(sample_ids)}")
        print(f"Trade ID format: {sample_ids[0].rsplit('_', 1)[0]}_<number>")
        
        # Trade ID input loop
        while True:
            print()
            trade_id = input("Enter trade ID (or 'back' to choose another file): ").strip()
            
            if trade_id.lower() in ('back', 'b', ''):
                break
            
            if trade_id.lower() in ('quit', 'q', 'exit'):
                print("\nGoodbye!")
                return
            
            # Find trade
            trade = get_trade_by_id(df, trade_id)
            if trade is None:
                # Try partial match
                matches = df[df['trade_id'].str.contains(trade_id, case=False)]
                if len(matches) == 1:
                    trade = matches.iloc[0]
                    print(f"   Found: {trade['trade_id']}")
                elif len(matches) > 1:
                    print(f"   Multiple matches: {matches['trade_id'].tolist()[:5]}")
                    continue
                else:
                    print(f"   Trade not found: {trade_id}")
                    continue
            
            # Fetch 1-minute bars
            symbol = trade['symbol']
            print(f"\nFetching bars for {symbol}...")
            bars_1min = fetch_bars_for_trade(
                symbol=symbol,
                entry_time=trade['entry_time'],
                exit_time=trade['exit_time']
            )
            
            if bars_1min is None or bars_1min.empty:
                print("   No bar data available. Skipping.")
                continue
            
            # Aggregate to 15-minute bars
            print("Aggregating to 15-minute bars...")
            bars_15min = aggregate_to_15min(bars_1min)
            print(f"   Created {len(bars_15min)} 15-min bars")
            
            # Detect FVGs on 15-minute timeframe
            print("Detecting FVGs on 15-minute timeframe...")
            fvgs = detect_fvgs_15min(bars_15min)
            long_fvgs = sum(1 for f in fvgs if f['type'] == 'long')
            short_fvgs = sum(1 for f in fvgs if f['type'] == 'short')
            print(f"   Found {len(fvgs)} FVGs ({long_fvgs} long, {short_fvgs} short)")
            
            # Calculate TP/SL levels
            print("Calculating TP/SL levels...")
            tpsl_levels = calculate_tpsl_levels(trade, bars_1min)
            
            # Create and show chart
            print("Creating chart...")
            fig = create_trade_chart(trade, bars_1min, bars_15min, fvgs, tpsl_levels)
            show_figure_with_hover_legend(fig)
            
            print(f"\n   Chart displayed for {trade['trade_id']}")


# =============================================================================
# MAIN INTERACTIVE LOOP - BACKTESTS
# =============================================================================

def select_backtest_csv_file(csv_files: List[Path]) -> Optional[Path]:
    """Interactive file selection for backtest files."""
    print("\n" + "="*60)
    print("AVAILABLE BACKTEST FILES (newest first)")
    print("="*60)
    
    for i, f in enumerate(csv_files, 1):
        # Get file info
        try:
            size_kb = f.stat().st_size / 1024
            mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            df = pd.read_csv(f, nrows=0)
            # Try to count rows without loading entire file
            with open(f, 'r') as fp:
                count = sum(1 for _ in fp) - 1  # Subtract header
        except:
            size_kb = 0
            mtime = "?"
            count = "?"
        
        print(f"  [{i}] {f.name}")
        print(f"      {count} trades | {size_kb:.0f} KB | {mtime}")
    
    print(f"\n  [0] Back to main menu")
    print()
    
    while True:
        try:
            choice = input("Select a file number: ").strip()
            if choice == '0' or choice.lower() in ('b', 'back'):
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(csv_files):
                return csv_files[idx]
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a number.")


def run_backtest_visualization():
    """Run backtest trade visualization loop."""
    csv_files = list_backtest_csv_files()
    if not csv_files:
        print("No backtest CSV files found.")
        return
    
    while True:
        # Select file
        selected_file = select_backtest_csv_file(csv_files)
        if selected_file is None:
            return
        
        print(f"\nLoading {selected_file.name}...")
        df = load_backtest_csv(selected_file)
        print(f"Loaded {len(df)} trades")
        
        # Symbol selection loop
        while True:
            # Get available symbols
            symbols = get_symbols_from_backtest(df)
            print(f"\nAvailable symbols ({len(symbols)}): {', '.join(symbols[:20])}")
            if len(symbols) > 20:
                print(f"   ... and {len(symbols) - 20} more")
            
            print()
            symbol_input = input("Enter symbol (or 'back' to choose another file): ").strip().upper()
            
            if symbol_input.lower() in ('back', 'b', ''):
                break
            
            if symbol_input.lower() in ('quit', 'q', 'exit'):
                print("\nGoodbye!")
                return
            
            if symbol_input not in symbols:
                print(f"   Symbol '{symbol_input}' not found in this backtest.")
                continue
            
            # Get trades for this symbol
            trades = get_trades_for_symbol(df, symbol_input)
            print(f"\n   Found {len(trades)} trades for {symbol_input}")
            
            # Show trade list
            print("\n" + "-"*80)
            print(f"TRADES FOR {symbol_input} (sorted by pattern timestamp ascending)")
            print("-"*80)
            
            for i, (idx, trade) in enumerate(trades.iterrows(), 1):
                direction = trade['direction']
                pattern_type = trade.get('pattern_type', 'FVG')
                # Use c1_timestamp, falling back to sweep_candle_timestamp for sweep patterns
                c1_ts = trade.get('c1_timestamp', trade.get('c1_datetime'))
                if pd.isna(c1_ts) or c1_ts is None:
                    c1_ts = trade.get('sweep_candle_timestamp', trade.get('sweep_candle_datetime', 'N/A'))
                entry_price = trade.get('entry_price', 0)
                exit_price = trade.get('exit_price', 0)
                pnl = trade.get('pnl_dollars', 0)
                win_loss = trade.get('win_loss', 'N/A')
                exit_reason = trade.get('exit_reason', 'N/A')
                
                pnl_str = f"+${pnl:.2f}" if pnl > 0 else f"${pnl:.2f}"
                dir_icon = "▲" if direction == 'long' else "▼"
                
                print(f"  [{i}] {dir_icon} {direction.upper():5s} | {pattern_type:15s} | {c1_ts}")
                print(f"       Entry: ${entry_price:.2f} → Exit: ${exit_price:.2f} | "
                      f"P&L: {pnl_str} ({win_loss}) | {exit_reason}")
            
            print(f"\n  [0] Back to symbol selection")
            print()
            
            # Trade selection loop
            while True:
                trade_choice = input("Select a trade number: ").strip()
                
                if trade_choice == '0' or trade_choice.lower() in ('b', 'back', ''):
                    break
                
                if trade_choice.lower() in ('quit', 'q', 'exit'):
                    print("\nGoodbye!")
                    return
                
                try:
                    trade_idx = int(trade_choice) - 1
                    if 0 <= trade_idx < len(trades):
                        trade = trades.iloc[trade_idx]
                    else:
                        print("Invalid selection. Try again.")
                        continue
                except ValueError:
                    print("Please enter a number.")
                    continue
                
                # Fetch 1-minute bars
                entry_time = trade.get('entry_datetime')
                exit_time = trade.get('exit_datetime')
                
                # Allow unfilled trades - check result field primarily, then fallback to entry/exit times
                trade_result_str = str(trade.get('result', '')).upper()
                is_unfilled_trade = (
                    trade_result_str in (
                        'NOT FILLED', 'UNFILLED', 'EXPIRED', 
                        'REVERSAL_CONFIRMATION_EXPIRED', 'REVERSAL_DEEP_RETRACE',
                        'PATTERN EXPIRED BEFORE ENTRY'
                    ) or pd.isna(entry_time) or pd.isna(exit_time)
                )
                if is_unfilled_trade:
                    print("   Trade is unfilled - using pattern timestamps for bar window")
                    # For unfilled trades, use pattern timestamp for time reference
                    pattern_ts = trade.get('c1_datetime') or trade.get('sweep_candle_datetime')
                    if pattern_ts is None or pd.isna(pattern_ts):
                        print("   Cannot determine pattern timestamp. Skipping.")
                        continue
                    # Set entry/exit to pattern timestamp for bar fetching
                    # The chart will show the pattern area even without an actual fill
                    entry_time = pattern_ts
                    exit_time = pattern_ts + timedelta(hours=2)  # Show 2 hours of bars
                    
                    # IMPORTANT: Update the trade dictionary so chart creation uses correct times
                    trade = trade.copy()  # Don't modify original
                    trade['entry_datetime'] = entry_time
                    trade['exit_datetime'] = exit_time
                    # Mark trade as unfilled for visualization purposes (used by chart creation)
                    trade['_viz_unfilled_trade'] = True
                    # Use planned entry price for display if entry_price is missing/invalid
                    if pd.isna(trade.get('entry_price')):
                        trade['entry_price'] = trade.get('entry', 0)
                    if pd.isna(trade.get('exit_price')):
                        trade['exit_price'] = trade.get('entry_price', 0)  # No exit, use entry
                
                print(f"\nFetching bars for {symbol_input}...")
                bars_1min = fetch_bars_for_trade(
                    symbol=symbol_input,
                    entry_time=entry_time,
                    exit_time=exit_time
                )
                
                if bars_1min is None or bars_1min.empty:
                    print("   No bar data available. Skipping.")
                    continue
                
                # Aggregate to 15-minute bars
                print("Aggregating to 15-minute bars...")
                bars_15min = aggregate_to_15min(bars_1min)
                print(f"   Created {len(bars_15min)} 15-min bars")
                
                # Detect FVGs on 15-minute timeframe
                print("Detecting FVGs on 15-minute timeframe...")
                fvgs = detect_fvgs_15min(bars_15min)
                long_fvgs = sum(1 for f in fvgs if f['type'] == 'long')
                short_fvgs = sum(1 for f in fvgs if f['type'] == 'short')
                print(f"   Found {len(fvgs)} FVGs ({long_fvgs} long, {short_fvgs} short)")
                
                # Calculate TP/SL levels from backtest data
                print("Calculating TP/SL levels from adjustment history...")
                tpsl_levels, stop_adjustments_list, tp_adjustments_list = calculate_tpsl_levels_backtest(trade, bars_1min)
                print(f"   Found {len(stop_adjustments_list)} SL adjustments, {len(tp_adjustments_list)} TP extensions")
                
                # Create and show chart
                print("Creating chart...")
                fig = create_backtest_trade_chart(
                    trade, bars_1min, bars_15min, fvgs, tpsl_levels,
                    stop_adjustments_list, tp_adjustments_list
                )
                show_figure_with_hover_legend(fig)
                
                print(f"\n   Chart displayed for {symbol_input} trade on {trade.get('backtest_date', 'N/A')}")

def build_backtest_trade_figure(trade: pd.Series) -> go.Figure:
    # --- Guardrails: ensure required functions exist in this module ---
    required = [
        "fetch_1min_bars_for_trade",
        "aggregate_to_15min",
        "detect_fvgs_15min",
        "calculate_tpsl_levels_backtest",
        "create_backtest_trade_chart",
    ]
    missing = [name for name in required if name not in globals()]
    if missing:
        raise NameError(
            "Missing required function(s) in viz.py scope: "
            + ", ".join(missing)
            + ".\n"
            "This usually means the function is named differently, or defined inside main()/if __name__ == '__main__'."
        )

    fetch_1min_bars_for_trade = globals()["fetch_1min_bars_for_trade"]
    aggregate_to_15min = globals()["aggregate_to_15min"]
    detect_fvgs_15min = globals()["detect_fvgs_15min"]
    calculate_tpsl_levels_backtest = globals()["calculate_tpsl_levels_backtest"]
    create_backtest_trade_chart = globals()["create_backtest_trade_chart"]

    # --- Determine times ---
    entry_time = trade.get("entry_datetime")
    exit_time = trade.get("exit_datetime")

    trade_result_str = str(trade.get("result", "")).upper()
    is_unfilled_trade = (
        trade_result_str in (
            "NOT FILLED", "UNFILLED", "EXPIRED",
            "REVERSAL_CONFIRMATION_EXPIRED", "REVERSAL_DEEP_RETRACE",
            "PATTERN EXPIRED BEFORE ENTRY",
        )
        or pd.isna(entry_time) or pd.isna(exit_time)
    )

    if is_unfilled_trade:
        pattern_ts = trade.get("c1_datetime") or trade.get("sweep_candle_datetime")
        if pattern_ts is None or pd.isna(pattern_ts):
            raise ValueError(
                "Unfilled trade: cannot determine pattern timestamp "
                "(c1_datetime / sweep_candle_datetime)."
            )
        entry_time = pattern_ts
        exit_time = pattern_ts

    # --- Fetch and build chart ---
    bars_1min = fetch_1min_bars_for_trade(
        symbol=trade.get("symbol"),
        entry_time=entry_time,
        exit_time=exit_time,
    )
    if bars_1min is None or bars_1min.empty:
        raise ValueError("No 1-minute bar data returned for this trade window.")

    bars_15min = aggregate_to_15min(bars_1min)
    fvgs = detect_fvgs_15min(bars_15min)

    tpsl_levels, stop_adjustments_list, tp_adjustments_list = calculate_tpsl_levels_backtest(
        trade, bars_1min
    )

    fig = create_backtest_trade_chart(
        trade,
        bars_1min,
        bars_15min,
        fvgs,
        tpsl_levels,
        stop_adjustments_list,
        tp_adjustments_list,
    )
    return fig


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main interactive loop."""
    print("\n" + "="*60)
    print("  TRADE VISUALIZATION TOOL")
    print("  View trade details from optimizer or backtests")
    print("="*60)
    
    while True:
        print("\n" + "-"*40)
        print("DATA SOURCE:")
        print("-"*40)
        print("  [1] Optimizer trade details")
        print("  [2] Backtest exports")
        print("  [0] Exit")
        print()
        
        choice = input("Select data source: ").strip()
        
        if choice == '0' or choice.lower() in ('q', 'quit', 'exit'):
            print("\nGoodbye!")
            break
        elif choice == '1':
            run_optimizer_visualization()
        elif choice == '2':
            run_backtest_visualization()
        else:
            print("Invalid selection. Please enter 1, 2, or 0.")


if __name__ == '__main__':
    main()
