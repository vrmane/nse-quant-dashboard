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
# STOCK MASTER
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
    "Sector",
    sorted(set(STOCK_MASTER.values())),
    default=sorted(set(STOCK_MASTER.values()))
)

view_mode = st.sidebar.radio("Timeframe", ["Daily", "Weekly", "Monthly"])

sma_short = st.sidebar.slider("SMA Short", 10, 50, 20)
sma_long = st.sidebar.slider("SMA Long", 50, 200, 50)
rsi_period = st.sidebar.slider("RSI Period", 7, 21, 14)

rolling_sharpe_window = st.sidebar.slider(
    "Rolling Sharpe Window",
    20, 252, 126
)

strategy_choice = st.sidebar.multiselect(
    "Strategies",
    ["Trend", "Mean Reversion", "Momentum"],
    default=["Trend"]
)

period_map = {
    "Daily": "3y",
    "Weekly": "7y",
    "Monthly": "15y"
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
    df["Momentum"] = close.pct_change(252)  # 12–1 style

    return df.dropna()

# ======================================================
# STRATEGIES
# ======================================================
def trend_strategy(df):
    return (
        (df["Close"] > df["SMA_S"]) &
        (df["SMA_S"] > df["SMA_L"]) &
        (df["RSI"] < 70)
    ).astype(int).to_numpy()

def mean_reversion_strategy(df):
    return (df["RSI"] < 30).astype(int).to_numpy()

def momentum_strategy(df):
    return (df["Momentum"] > 0).astype(int).to_numpy()

# ======================================================
# PERFORMANCE & STATS
# ======================================================
def compute_performance(df, position):
    returns = df["Close"].pct_change().fillna(0)
    strat_returns = returns * np.roll(position, 1)

    equity = (1 + strat_returns).cumprod()
    drawdown = equity / equity.cummax() - 1

    sharpe = strat_returns.mean() / strat_returns.std() * np.sqrt(252)
    downside = strat_returns[strat_returns < 0]
    sortino = (
        strat_returns.mean() / downside.std() * np.sqrt(252)
        if len(downside) > 0 else np.nan
    )

    rolling_sharpe = (
        strat_returns.rolling(rolling_sharpe_window).mean() /
        strat_returns.rolling(rolling_sharpe_window).std()
    ) * np.sqrt(252)

    return equity, drawdown, sharpe, sortino, rolling_sharpe, strat_returns

def trade_stats(strat_returns):
    trades = strat_returns[strat_returns != 0]

    if trades.empty:
        return {
            "Trades": 0,
            "Win Rate %": 0,
            "Expectancy %": 0
        }

    wins = trades[trades > 0]
    losses = trades[trades < 0]

    win_rate = len(wins) / len(trades) * 100
    expectancy = trades.mean() * 100

    return {
        "Trades": len(trades),
        "Win Rate %": round(win_rate, 2),
        "Expectancy %": round(expectancy, 2)
    }

# ======================================================
# PLOTS
# ======================================================
def plot_equity(equity, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity)
    ax.set_title(title)
    st.pyplot(fig)

def plot_drawdown(drawdown):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(drawdown.index, drawdown, color="red", alpha=0.4)
    ax.set_title("Rolling Drawdown")
    st.pyplot(fig)

def plot_rolling_sharpe(rolling_sharpe):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(rolling_sharpe, color="purple")
    ax.axhline(0, linestyle="--", color="black")
    ax.set_title("Rolling Sharpe Ratio")
    st.pyplot(fig)

# ======================================================
# DASHBOARD
# ======================================================
st.title("📈 NSE Quant Research Dashboard")

stocks = [s for s, sec in STOCK_MASTER.items() if sec in selected_sectors]
screener_rows = []

for stock in stocks:
    df = yf.download(stock, period=period_map[view_mode], progress=False)
    if df.empty:
        continue

    df = clean_ohlc(df)
    df = resample_data(df, view_mode)
    df = compute_indicators(df)

    st.subheader(stock)

    for strat in strategy_choice:
        if strat == "Trend":
            pos = trend_strategy(df)
        elif strat == "Mean Reversion":
            pos = mean_reversion_strategy(df)
        else:
            pos = momentum_strategy(df)

        equity, dd, sharpe, sortino, rsharpe, sret = compute_performance(df, pos)
        stats = trade_stats(sret)

        st.markdown(f"### 🧠 {strat} Strategy")

        c1, c2, c3 = st.columns(3)
        c1.metric("Sharpe", round(sharpe, 2))
        c2.metric("Sortino", round(sortino, 2))
        c3.metric("Trades", stats["Trades"])

        plot_equity(equity, f"{stock} – {strat} Equity Curve")
        plot_drawdown(dd)
        plot_rolling_sharpe(rsharpe)

        screener_rows.append({
            "Stock": stock,
            "Strategy": strat,
            "Sharpe": round(sharpe, 2),
            "Sortino": round(sortino, 2),
            "Trades": stats["Trades"],
            "Win Rate %": stats["Win Rate %"],
            "Expectancy %": stats["Expectancy %"]
        })

# ======================================================
# SCREENER TABLE
# ======================================================
st.subheader("🧾 Strategy Screener")
screener_df = pd.DataFrame(screener_rows)
st.dataframe(screener_df, use_container_width=True)

# ======================================================
# EXPORT
# ======================================================
output = BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    screener_df.to_excel(writer, index=False, sheet_name="Strategies")

output.seek(0)

st.download_button(
    "⬇️ Download Strategy Screener",
    data=output,
    file_name="nse_strategy_screener.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
