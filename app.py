import os
from pathlib import Path
import streamlit as st

# Import your existing script as a module
import scripts.viz

st.set_page_config(page_title="Trading Algo Demo", layout="wide")

st.title("Trading Algorithm Demo")
st.caption("Select a backtest file + a trade and render the interactive Plotly chart.")

# ---- Secrets -> env vars (so viz.py can keep using os.getenv) ----
# In Community Cloud, you'll set these in the Secrets UI (Step 7).
if "ALPACA_API_KEY" in st.secrets:
    os.environ["ALPACA_API_KEY"] = st.secrets["ALPACA_API_KEY"]
if "ALPACA_SECRET_KEY" in st.secrets:
    os.environ["ALPACA_SECRET_KEY"] = st.secrets["ALPACA_SECRET_KEY"]

# ---- Locate backtest CSVs exactly where viz.py expects them ----
project_root = Path(__file__).parent
backtests_dir = project_root / "outputs" / "backtests"
csv_files = sorted(backtests_dir.glob("*.csv"))

if not csv_files:
    st.error(f"No CSV files found in {backtests_dir}")
    st.stop()

csv_choice = st.selectbox("Backtest CSV", csv_files, format_func=lambda p: p.name)

df = viz.load_backtest_csv(csv_choice)

# Choose symbol if present
if "symbol" in df.columns:
    symbols = sorted(df["symbol"].dropna().unique().tolist())
    symbol = st.selectbox("Symbol", symbols)
    df_view = df[df["symbol"] == symbol].copy()
else:
    df_view = df.copy()

# Choose trade_id if present
if "trade_id" in df_view.columns:
    trade_ids = df_view["trade_id"].dropna().unique().tolist()
    trade_id = st.selectbox("Trade ID", trade_ids)
    trade = viz.get_trade_by_id(df_view, trade_id)
    if trade is None:
        st.warning("Trade not found.")
        st.stop()
else:
    st.error("This CSV doesn't have a trade_id column.")
    st.stop()

# Render chart (your viz.py likely has a chart builder; common names below)
# If your function name differs, update this one line.
fig = viz.build_trade_figure(trade, df_view) if hasattr(viz, "build_trade_figure") else None

if fig is None:
    st.error("Couldn't find a figure builder in viz.py. Search for the function that returns a Plotly Figure and call it here.")
    st.stop()

st.plotly_chart(fig, use_container_width=True)
