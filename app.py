from pathlib import Path
import os
import streamlit as st
from scripts import viz
import pandas as pd

st.set_page_config(page_title="Trading Algo Demo", layout="wide")

st.write("viz.VIZ_VERSION =", getattr(viz, "VIZ_VERSION", "MISSING"))
st.write("has fetch_1min_bars_for_trade =", hasattr(viz, "fetch_1min_bars_for_trade"))
st.write("viz module path =", getattr(viz, "__file__", "unknown"))



# -----------------------------------------------------------------------------
# Secrets -> env vars (so viz.py can keep using os.getenv)
# -----------------------------------------------------------------------------
if "ALPACA_API_KEY" in st.secrets:
    os.environ["ALPACA_API_KEY"] = st.secrets["ALPACA_API_KEY"]
if "ALPACA_SECRET_KEY" in st.secrets:
    os.environ["ALPACA_SECRET_KEY"] = st.secrets["ALPACA_SECRET_KEY"]

st.title("Trading Algorithm Demo")
st.caption("Pick a backtest CSV, select a symbol + trade, and view the Plotly chart.")

# -----------------------------------------------------------------------------
# Locate CSV files (use viz.BACKTESTS_DIR so paths stay consistent)
# -----------------------------------------------------------------------------
backtests_dir = Path(viz.BACKTESTS_DIR).resolve()

with st.expander("Debug: filesystem paths", expanded=False):
    st.write("viz.BACKTESTS_DIR:", str(backtests_dir))
    st.write("Exists:", backtests_dir.exists())
    if backtests_dir.exists():
        st.write("All files:", [p.name for p in backtests_dir.glob("*")])
        st.write("CSV files:", [p.name for p in backtests_dir.glob("*.csv")])

csv_files = sorted(backtests_dir.glob("*.csv"))
if not csv_files:
    st.error(f"No CSV files found in {backtests_dir}")
    st.stop()

csv_path = st.selectbox("Backtest CSV", csv_files, format_func=lambda p: p.name)

# -----------------------------------------------------------------------------
# Load the CSV using your existing loader
# -----------------------------------------------------------------------------
df = viz.load_backtest_csv(csv_path)

st.write(f"Loaded **{len(df)}** rows from `{csv_path.name}`")

# -----------------------------------------------------------------------------
# Symbol selection (if present)
# -----------------------------------------------------------------------------
if "symbol" in df.columns:
    symbols = sorted(df["symbol"].dropna().unique().tolist())
    symbol = st.selectbox("Symbol", symbols)
    df_symbol = df[df["symbol"] == symbol].copy()
else:
    st.warning("No `symbol` column found. Showing all trades.")
    df_symbol = df.copy()

# -----------------------------------------------------------------------------
# Trade selection
# Use viz.get_trades_for_symbol if available, else fallback to the dataframe
# -----------------------------------------------------------------------------
if hasattr(viz, "get_trades_for_symbol") and "symbol" in df.columns:
    trades_df = viz.get_trades_for_symbol(df, symbol)  # expected to return a DF sorted
else:
    trades_df = df_symbol

if trades_df is None or trades_df.empty:
    st.error("No trades found for that selection.")
    st.stop()

# Make a display label list
def trade_label(row: pd.Series) -> str:
    direction = str(row.get("direction", "n/a")).upper()
    pattern = str(row.get("pattern_type", row.get("pattern", "FVG")))
    # try to display something timestamp-y
    ts = row.get("c1_timestamp", row.get("c1_datetime", row.get("sweep_candle_timestamp", row.get("sweep_candle_datetime", ""))))
    pnl = row.get("pnl_dollars", row.get("pnl", ""))
    try:
        pnl_str = f"{float(pnl):+.2f}"
    except Exception:
        pnl_str = str(pnl)
    return f"{direction:5s} | {pattern:15s} | {ts} | PnL {pnl_str}"

trade_indices = list(trades_df.index)

selected_idx = st.selectbox(
    "Trade",
    trade_indices,
    format_func=lambda idx: trade_label(trades_df.loc[idx])
)

trade = trades_df.loc[selected_idx]

# Show trade details quickly
with st.expander("Trade details", expanded=False):
    st.dataframe(trade.to_frame("value"))

# -----------------------------------------------------------------------------
# Build and display chart
# This calls a new helper we’ll add to viz.py below: build_backtest_trade_figure()
# -----------------------------------------------------------------------------
st.subheader("Chart")

try:
    with st.spinner(
        "Building chart (fetching bars, aggregating, detecting FVGs, calculating TP/SL levels)..."
    ):
        fig = viz.build_backtest_trade_figure(trade)

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.exception(e)
    st.stop()

