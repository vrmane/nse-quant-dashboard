import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="NSE Quant Dashboard", layout="wide")

# =========================
# STOCK MASTER
# =========================
STOCK_MASTER = {
    "RELIANCE.NS": "Energy",
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "HDFCBANK.NS": "Banking",
    "ICICIBANK.NS": "Banking",
    "LT.NS": "Infra"
}

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Controls")

selected_sectors = st.sidebar.multiselect(
    "Sector",
    sorted(set(STOCK_MASTER.values())),
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

# =========================
# HELPERS
# =========================
def clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make yfinance output 100% predictable:
    - Flatten columns
    - Ensure Close is 1D float Series
    """
    df = df.copy()

    # Flatten multi-index columns if present
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
    df = df.copy()
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
# SIGNALS (NUMPY-LEVEL SAFE)
# =========================
def generate_positions(df):
    close = df["Close"].to_numpy()
    sma_s = df["SMA_S"].to_numpy()
    sma_l = df["SMA_L"].to_numpy()
    rsi = df["RSI"].to_numpy()

    return ((close > sma_s) & (sma_s > sma_l) & (rsi < 70)).astype(int)

def compute_metrics(df):
    df = df.copy()

    df["Position"] = generate_positions(df)
    df["Returns"] = df["Close"].pct_change().fillna(0)
    df["Strategy_Return"] = df["Returns"] * np.roll(df["Position"], 1)

    equity = (1 + df["Strategy_Return"]).cumprod()
    max_dd = ((equity / equity.cummax()) - 1).min()
    vol = df["Strategy_Return"].std() * np.sqrt(252)

    return equity, round(max_dd * 100, 2), round(vol * 100, 2)

def signal_label(row):
    if row.Close > row.SMA_S > row.SMA_L and row.RSI < 70:
        return "Bullish"
    if row.Close < row.SMA_S < row.SMA_L and row.RSI <= 30:
        return "Bearish Oversold"
    if row.Close < row.SMA_S < row.SMA_L:
        return "Bearish"
    return "Neutral"

def plot_equity(equity, stock):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity, label="Strategy Equity Curve")
    ax.set_title(f"{stock} – Cumulative Returns")
    ax.legend()
    st.pyplot(fig)

# =========================
# DASHBOARD
# =========================
st.title("📈 NSE Quant Dashboard")

stocks = [s for s, sec in STOCK_MASTER.items() if sec in selected_sectors]
results = []

for stock in stocks:
    df = yf.download(stock, period=period_map[view_mode], progress=False)

    if df.empty:
        continue

    df = clean_ohlc(df)
    df = resample_data(df, view_mode)
    df = compute_indicators(df)

    equity, max_dd, vol = compute_metrics(df)
    last = df.iloc[-1]

    sig = signal_label(last)

    results.append({
        "Stock": stock,
        "Sector": STOCK_MASTER[stock],
        "Last Close": round(float(last.Close), 2),
        "RSI": round(float(last.RSI), 2),
        "Signal": sig,
        "Max Drawdown %": max_dd,
        "Volatility %": vol
    })

    st.subheader(stock)
    c1, c2, c3 = st.columns(3)
    c1.metric("Signal", sig)
    c2.metric("Max Drawdown", f"{max_dd}%")
    c3.metric("Volatility", f"{vol}%")

    plot_equity(equity, stock)

# =========================
# SCREENER
# =========================
st.subheader("🧾 Sector-wise Screener")
screener_df = pd.DataFrame(results)
st.dataframe(screener_df, use_container_width=True)

st.download_button(
    "⬇️ Download Screener (Excel)",
    data=screener_df.to_excel(index=False),
    file_name="nse_quant_screener.xlsx"
)
