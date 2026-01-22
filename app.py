import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="NSE Quant Dashboard",
    layout="wide"
)

# =========================
# STOCK MASTER (SECTORS)
# =========================
STOCK_MASTER = {
    "RELIANCE.NS": "Energy",
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "HDFCBANK.NS": "Banking",
    "ICICIBANK.NS": "Banking",
    "LT.NS": "Infra"
}

BENCHMARK = "^NSEI"

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.title("⚙️ Controls")

selected_sectors = st.sidebar.multiselect(
    "Sector Filter",
    options=sorted(set(STOCK_MASTER.values())),
    default=sorted(set(STOCK_MASTER.values()))
)

view_mode = st.sidebar.radio(
    "Timeframe",
    ["Daily", "Weekly", "Monthly"]
)

sma_short = st.sidebar.slider("SMA Short", 10, 50, 20)
sma_long = st.sidebar.slider("SMA Long", 50, 200, 50)
rsi_period = st.sidebar.slider("RSI Period", 7, 21, 14)

period_map = {
    "Daily": "2y",
    "Weekly": "5y",
    "Monthly": "10y"
}

# =========================
# DATA FUNCTIONS
# =========================
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

# =========================
# VECTORISED SIGNAL LOGIC (SAFE)
# =========================
def generate_positions(df):
    """
    Vectorized bullish signal:
    Close > SMA short > SMA long AND RSI < 70
    """
    return (
        (df["Close"] > df["SMA_S"]) &
        (df["SMA_S"] > df["SMA_L"]) &
        (df["RSI"] < 70)
    ).astype(int)

# =========================
# METRICS
# =========================
def compute_metrics(df):
    df = df.copy()

    df["Position"] = generate_positions(df)
    df["Returns"] = df["Close"].pct_change()
    df["Strategy_Return"] = df["Returns"] * df["Position"].shift(1)

    equity_curve = (1 + df["Strategy_Return"].fillna(0)).cumprod()

    max_drawdown = ((equity_curve / equity_curve.cummax()) - 1).min()
    volatility = df["Strategy_Return"].std() * np.sqrt(252)

    return equity_curve, round(max_drawdown * 100, 2), round(volatility * 100, 2)

def get_signal_label(last_row):
    if (
        last_row.Close > last_row.SMA_S and
        last_row.SMA_S > last_row.SMA_L and
        last_row.RSI < 70
    ):
        return "Bullish"
    elif (
        last_row.Close < last_row.SMA_S and
        last_row.SMA_S < last_row.SMA_L and
        last_row.RSI <= 30
    ):
        return "Bearish Oversold"
    elif last_row.Close < last_row.SMA_S < last_row.SMA_L:
        return "Bearish"
    else:
        return "Neutral"

# =========================
# PLOTS
# =========================
def plot_equity_curve(curve, stock):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(curve, label="Strategy Equity Curve")
    ax.set_title(f"{stock} – Cumulative Returns")
    ax.legend()
    st.pyplot(fig)

# =========================
# FILTER STOCKS
# =========================
stocks_to_scan = [
    s for s, sector in STOCK_MASTER.items()
    if sector in selected_sectors
]

# =========================
# DASHBOARD
# =========================
st.title("📈 NSE Quant Dashboard")

results = []

for stock in stocks_to_scan:
    df = yf.download(
        stock,
        period=period_map[view_mode],
        progress=False
    )

    if df.empty:
        continue

    df = resample_data(df, view_mode)
    df = compute_indicators(df)

    equity, max_dd, vol = compute_metrics(df)
    last = df.iloc[-1]

    signal = get_signal_label(last)

    results.append({
        "Stock": stock,
        "Sector": STOCK_MASTER[stock],
        "Last Close": round(float(last.Close), 2),
        "RSI": round(float(last.RSI), 2),
        "Signal": signal,
        "Max Drawdown %": max_dd,
        "Volatility %": vol
    })

    st.subheader(stock)
    col1, col2, col3 = st.columns(3)
    col1.metric("Signal", signal)
    col2.metric("Max Drawdown", f"{max_dd}%")
    col3.metric("Volatility", f"{vol}%")

    plot_equity_curve(equity, stock)

# =========================
# SCREENER TABLE
# =========================
st.subheader("🧾 Sector-wise Screener")
screener_df = pd.DataFrame(results)
st.dataframe(screener_df, use_container_width=True)

# =========================
# EXPORT
# =========================
st.download_button(
    "⬇️ Download Screener (Excel)",
    data=screener_df.to_excel(index=False),
    file_name="nse_quant_screener.xlsx"
)
