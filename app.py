import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="NSE Quant Dashboard", layout="wide")

# =========================
# STOCK MASTER (WITH SECTORS)
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

selected_sector = st.sidebar.multiselect(
    "Select Sector",
    options=sorted(set(STOCK_MASTER.values())),
    default=sorted(set(STOCK_MASTER.values()))
)

view_mode = st.sidebar.radio("Time View", ["Daily", "Weekly", "Monthly"])

sma_short = st.sidebar.slider("SMA Short", 10, 50, 20)
sma_long = st.sidebar.slider("SMA Long", 50, 200, 50)
rsi_period = st.sidebar.slider("RSI Period", 7, 21, 14)

period_map = {
    "Daily": "2y",
    "Weekly": "5y",
    "Monthly": "10y"
}

# =========================
# FUNCTIONS
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

def generate_signal(row):
    if row.Close > row.SMA_S > row.SMA_L and row.RSI < 70:
        return 1
    return 0

def compute_metrics(df):
    df = df.copy()
    df["Position"] = df.apply(generate_signal, axis=1)
    df["Returns"] = df["Close"].pct_change()
    df["Strategy_Return"] = df["Returns"] * df["Position"].shift(1)

    cumulative = (1 + df["Strategy_Return"].fillna(0)).cumprod()
    max_drawdown = ((cumulative / cumulative.cummax()) - 1).min()

    volatility = df["Strategy_Return"].std() * np.sqrt(252)

    return cumulative, round(max_drawdown * 100, 2), round(volatility * 100, 2)

def plot_cumulative(cumulative):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cumulative, label="Strategy Equity Curve")
    ax.set_title("Cumulative Strategy Returns")
    ax.legend()
    st.pyplot(fig)

# =========================
# FILTER STOCKS BY SECTOR
# =========================
filtered_stocks = [
    s for s, sector in STOCK_MASTER.items()
    if sector in selected_sector
]

# =========================
# MAIN DASHBOARD
# =========================
st.title("📈 NSE Quant Dashboard")

results = []

for stock in filtered_stocks:
    df = yf.download(stock, period=period_map[view_mode], progress=False)
    if df.empty:
        continue

    df = resample_data(df, view_mode)
    df = compute_indicators(df)

    cumulative, max_dd, vol = compute_metrics(df)

    results.append({
        "Stock": stock,
        "Sector": STOCK_MASTER[stock],
        "Max Drawdown %": max_dd,
        "Volatility %": vol
    })

    st.subheader(stock)
    col1, col2 = st.columns(2)
    col1.metric("Max Drawdown", f"{max_dd}%")
    col2.metric("Volatility", f"{vol}%")

    plot_cumulative(cumulative)

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
    "⬇️ Download Screener",
    screener_df.to_excel(index=False),
    file_name="nse_quant_screener.xlsx"
)
