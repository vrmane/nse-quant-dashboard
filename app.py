import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="NSE Quant Dashboard", layout="wide")

# ======================================================
# STOCK MASTER WITH SECTORS
# ======================================================
STOCK_MASTER = {
    "RELIANCE.NS": "Energy",
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "HDFCBANK.NS": "Banking",
    "ICICIBANK.NS": "Banking",
    "LT.NS": "Infra"
}

# ======================================================
# SIDEBAR CONTROLS
# ======================================================
st.sidebar.title("⚙️ Controls")

selected_sectors = st.sidebar.multiselect(
    "Sector Filter",
    options=sorted(set(STOCK_MASTER.values())),
    default=sorted(set(STOCK_MASTER.values()))
)

view_mode = st.sidebar.radio("Timeframe", ["Daily", "Weekly", "Monthly"])

sma_short = st.sidebar.slider("SMA Short", 10, 50, 20)
sma_long = st.sidebar.slider("SMA Long", 50, 200, 50)
rsi_period = st.sidebar.slider("RSI Period", 7, 21, 14)

period_map = {
    "Daily": "2y",
    "Weekly": "5y",
    "Monthly": "10y"
}

# ======================================================
# DATA PREP
# ======================================================
def clean_ohlc(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    return df.dropna(subset=["Close"])

def resample_data(df, mode):
    if mode == "Weekly":
        return df.resample("W").last()
    if mode == "Monthly":
        return df.resample("M").last()
    return df

def compute_indicators(df):
    close = df["Close"]
    df["SMA_S"] = close.rolling(sma_short).mean()
    df["SMA_L"] = close.rolling(sma_long).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))
    return df.dropna()

# ======================================================
# STRATEGIES
# ======================================================
def trend_strategy(df):
    close = df["Close"].to_numpy()
    sma_s = df["SMA_S"].to_numpy()
    sma_l = df["SMA_L"].to_numpy()
    rsi = df["RSI"].to_numpy()

    position = ((close > sma_s) & (sma_s > sma_l) & (rsi < 70)).astype(int)
    return position

def buy_and_hold(df):
    return np.ones(len(df))

# ======================================================
# METRICS
# ======================================================
def compute_performance(df, position):
    returns = df["Close"].pct_change().fillna(0)
    strat_returns = returns * np.roll(position, 1)

    equity = (1 + strat_returns).cumprod()

    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()

    sharpe = (strat_returns.mean() / strat_returns.std()) * np.sqrt(252)
    downside = strat_returns[strat_returns < 0]
    sortino = (
        strat_returns.mean() /
        downside.std()
    ) * np.sqrt(252) if len(downside) > 0 else np.nan

    return {
        "equity": equity,
        "returns": strat_returns,
        "drawdown": drawdown,
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2),
        "MaxDD": round(max_dd * 100, 2)
    }

# ======================================================
# PLOTS
# ======================================================
def plot_equity_comparison(trend_eq, bh_eq, stock):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(trend_eq, label="Trend Strategy")
    ax.plot(bh_eq, label="Buy & Hold", linestyle="--")
    ax.set_title(f"{stock} – Equity Curve Comparison")
    ax.legend()
    st.pyplot(fig)

def plot_drawdown(drawdown, stock):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(drawdown.index, drawdown, color="red", alpha=0.4)
    ax.set_title(f"{stock} – Rolling Drawdown")
    st.pyplot(fig)

# ======================================================
# DASHBOARD
# ======================================================
st.title("📈 NSE Quant Dashboard")

stocks_to_scan = [
    s for s, sec in STOCK_MASTER.items()
    if sec in selected_sectors
]

results = []

for stock in stocks_to_scan:
    df = yf.download(stock, period=period_map[view_mode], progress=False)
    if df.empty:
        continue

    df = clean_ohlc(df)
    df = resample_data(df, view_mode)
    df = compute_indicators(df)

    trend_pos = trend_strategy(df)
    bh_pos = buy_and_hold(df)

    trend_perf = compute_performance(df, trend_pos)
    bh_perf = compute_performance(df, bh_pos)

    last = df.iloc[-1]

    results.append({
        "Stock": stock,
        "Sector": STOCK_MASTER[stock],
        "Sharpe (Trend)": trend_perf["Sharpe"],
        "Sortino (Trend)": trend_perf["Sortino"],
        "Max DD % (Trend)": trend_perf["MaxDD"],
        "Sharpe (B&H)": bh_perf["Sharpe"]
    })

    st.subheader(stock)

    c1, c2, c3 = st.columns(3)
    c1.metric("Sharpe", trend_perf["Sharpe"])
    c2.metric("Sortino", trend_perf["Sortino"])
    c3.metric("Max Drawdown", f'{trend_perf["MaxDD"]}%')

    plot_equity_comparison(
        trend_perf["equity"],
        bh_perf["equity"],
        stock
    )

    plot_drawdown(trend_perf["drawdown"], stock)

# ======================================================
# SCREENER TABLE
# ======================================================
st.subheader("🧾 Strategy Comparison Screener")
screener_df = pd.DataFrame(results)
st.dataframe(screener_df, use_container_width=True)

# ======================================================
# EXPORT
# ======================================================
output = BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    screener_df.to_excel(writer, index=False, sheet_name="Strategy_Comparison")

output.seek(0)

st.download_button(
    label="⬇️ Download Strategy Screener",
    data=output,
    file_name="nse_strategy_comparison.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
