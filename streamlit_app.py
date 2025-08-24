import streamlit as st
import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
from io import StringIO

# Try Plotly; fall back if not present
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px, go = None, None

# -----------------------------
# App Setup
# -----------------------------
st.set_page_config(page_title="📈 Indian Stock Tracker ", page_icon="🇮🇳", layout="wide")
st.title("📈 S.H.U.B.H. India Stock Tracker")
st.caption("Smart Hub for Understanding Business Holdings (India)")
st.write("Track **NSE/BSE** stocks with Yahoo Finance data, compare performance, analyze risk, and download results.")

# -----------------------------
# Helpers
# -----------------------------
def add_suffix(symbol: str, sfx: str) -> str:
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol
    return f"{symbol}{sfx}"

@st.cache_data(show_spinner=False)
def fetch_history(tickers_full, start, end, interval="1d"):
    """Download adjusted close for each ticker into a single DataFrame."""
    frames = []
    for full in tickers_full:
        try:
            hist = yf.Ticker(full).history(start=start, end=end, interval=interval, auto_adjust=True, actions=False)
            if not hist.empty and "Close" in hist.columns:
                root = full.replace(".NS", "").replace(".BO", "")
                frames.append(hist["Close"].rename(root))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, axis=1).sort_index()
    data.index.name = "Date"
    return data

@st.cache_data(show_spinner=False)
def fast_fundamentals(full_symbol: str):
    """Lightweight fundamentals from yfinance fast_info (cached)."""
    try:
        t = yf.Ticker(full_symbol)
        fi = getattr(t, "fast_info", {})
        # 52w metrics may be None depending on ticker support
        return {
            "currency": getattr(fi, "currency", None) if hasattr(fi, "currency") else (fi.get("currency") if isinstance(fi, dict) else None),
            "market_cap": getattr(fi, "market_cap", None) if hasattr(fi, "market_cap") else (fi.get("market_cap") if isinstance(fi, dict) else None),
            "year_high": getattr(fi, "year_high", None) if hasattr(fi, "year_high") else (fi.get("year_high") if isinstance(fi, dict) else None),
            "year_low": getattr(fi, "year_low", None) if hasattr(fi, "year_low") else (fi.get("year_low") if isinstance(fi, dict) else None),
            "last_price": getattr(fi, "last_price", None) if hasattr(fi, "last_price") else (fi.get("last_price") if isinstance(fi, dict) else None),
        }
    except Exception:
        return {}

def compute_indicators(prices: pd.Series) -> pd.DataFrame:
    df = prices.to_frame("Close").copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    # RSI(14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan))
    df["RSI14"] = 100 - (100 / (1 + rs))
    return df

def perf_stats(series: pd.Series, freq_per_year=252):
    """Return, CAGR, vol, max DD, Sharpe (rf=0)."""
    series = series.dropna()
    if series.empty:
        return None
    ret = series.pct_change().dropna()
    # CAGR
    days = (series.index[-1] - series.index[0]).days
    if days <= 0:
        cagr = np.nan
    else:
        cagr = (series.iloc[-1] / series.iloc[0]) ** (365.25 / days) - 1
    vol = ret.std() * np.sqrt(freq_per_year)
    # Max Drawdown
    roll_max = series.cummax()
    dd = series / roll_max - 1.0
    max_dd = dd.min()
    sharpe = (ret.mean() * freq_per_year) / vol if vol and not np.isnan(vol) and vol != 0 else np.nan
    total = series.iloc[-1] / series.iloc[0] - 1
    return {"Return": total, "CAGR": cagr, "Volatility": vol, "MaxDD": max_dd, "Sharpe": sharpe}

def to_pct(x):
    return f"{x*100:,.2f}%" if pd.notnull(x) else "—"

def resample_prices(df: pd.DataFrame, mode: str):
    if mode == "Daily":
        return df
    if mode == "Weekly":
        return df.resample("W-FRI").last()
    if mode == "Monthly":
        return df.resample("M").last()
    return df

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("📊 Stock Selection")

# Exchange toggle
exchange = st.sidebar.radio("Exchange", ["NSE (.NS)", "BSE (.BO)"], index=0)
suffix = ".NS" if "NSE" in exchange else ".BO"

# Built-in populars
POPULAR = [
    "RELIANCE","TCS","HDFCBANK","ICICIBANK","INFY","KOTAKBANK","SBIN",
    "LT","ITC","HINDUNILVR","BHARTIARTL","ASIANPAINT","AXISBANK","MARUTI",
    "BAJFINANCE","WIPRO","NTPC","POWERGRID","ULTRACEMCO","HCLTECH",
    "TATAMOTORS","TATASTEEL","ADANIENT","ADANIPORTS","SUNPHARMA",
    "ONGC","COALINDIA","INDUSINDBK","NESTLEIND","BAJAJFINSV"
]

# Presets from yfinance (NIFTY)
with st.sidebar.expander("⭐ Quick Presets"):
    colp1, colp2 = st.columns(2)
    preset_choice = None
    if colp1.button("NIFTY 50"):
        try:
            preset_choice = [s.replace(".NS","") for s in yf.tickers_nifty50()]
        except Exception:
            st.warning("Could not fetch NIFTY 50. Using popular list.")
            preset_choice = POPULAR
    if colp2.button("NIFTY BANK"):
        try:
            preset_choice = [s.replace(".NS","") for s in yf.tickers_niftybank()]
        except Exception:
            st.warning("Could not fetch NIFTY BANK.")

# Upload master list (to include ALL listed companies)
with st.sidebar.expander("📥 Import Full Symbol List (Optional)"):
    st.write("Upload a CSV with columns: **symbol** (required), optional **name**, **exchange**.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    url_input = st.text_input("...or paste CSV URL (raw GitHub/CSV endpoint)", value="")
    imported_symbols = []
    if uploaded is not None:
        try:
            df_imp = pd.read_csv(uploaded)
            if "symbol" in df_imp.columns:
                imported_symbols = [str(s).upper() for s in df_imp["symbol"].dropna().unique().tolist()]
            else:
                st.error("CSV must contain a 'symbol' column.")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
    elif url_input.strip():
        try:
            # NOTE: pandas can read many HTTP CSVs; this will run only when the app is deployed.
            df_imp = pd.read_csv(url_input.strip())
            if "symbol" in df_imp.columns:
                imported_symbols = [str(s).upper() for s in df_imp["symbol"].dropna().unique().tolist()]
            else:
                st.error("Remote CSV must contain a 'symbol' column.")
        except Exception as e:
            st.error(f"Failed to fetch CSV from URL: {e}")

# Main selection list
base_options = sorted(set(POPULAR + (preset_choice or []) + imported_symbols))
if not base_options:
    base_options = POPULAR

DEFAULT_TICKERS = ["RELIANCE","TCS","HDFCBANK","ICICIBANK"]
if preset_choice:
    DEFAULT_TICKERS = preset_choice[:6] if len(preset_choice) >= 6 else preset_choice

tickers_root = st.sidebar.multiselect(
    "Select Stocks to Compare",
    options=base_options,
    default=[t for t in DEFAULT_TICKERS if t in base_options],
    help="Symbols without suffix; app adds .NS/.BO automatically."
)

# Free-text add
extra = st.sidebar.text_input("Add more (comma-separated, e.g., INFY, SBIN)", value="")
if extra.strip():
    tickers_root.extend([e.strip().upper() for e in extra.split(",") if e.strip()])

# De-duplicate while preserving order
seen = set()
tickers_root = [x for x in tickers_root if not (x in seen or seen.add(x))]

# Dates, interval & resampling
start_date = st.sidebar.date_input("Start Date", dt.date(2024, 1, 1))
end_date = st.sidebar.date_input("End Date", dt.date.today())
interval = st.sidebar.selectbox("Data Interval", ["1d","1wk","1mo"], index=0)
resample_mode = st.sidebar.selectbox("Resample To", ["Daily","Weekly","Monthly"], index=0)

# Portfolio weights setting
with st.sidebar.expander("🎯 Portfolio Weights"):
    weight_mode = st.radio("Weighting", ["Equal Weight", "Custom Weights"], index=0)
    custom_weights = {}
    if weight_mode == "Custom Weights" and tickers_root:
        for t in tickers_root:
            custom_weights[t] = st.slider(f"{t} %", 0, 100, 0, step=1)
        total_w = sum(custom_weights.values())
        st.caption(f"Total: **{total_w}%** (auto-normalized if ≠ 100%)")

# -----------------------------
# Main
# -----------------------------
if start_date > end_date:
    st.error("⚠️ Start date must be before end date.")
elif not tickers_root:
    st.warning("Please select at least one stock.")
else:
    suffix_label = "NSE" if suffix == ".NS" else "BSE"
    full = [add_suffix(t, suffix) for t in tickers_root]

    with st.spinner(f"Fetching data from Yahoo Finance ({suffix_label})..."):
        data_raw = fetch_history(full, start_date, end_date, interval=interval)

    if data_raw.empty:
        st.error("No data returned. Try different symbols, exchange, or date range.")
    else:
        # Optional resampling
        data = resample_prices(data_raw, resample_mode)

        # Toggle raw preview
        with st.expander("📄 Raw Data Preview"):
            st.dataframe(data.tail(20))

        # ----------------- Price chart with SMAs/RSI per selected (single or multiple) -----------------
        st.subheader(f"💹 Price Chart ({suffix_label}) — {resample_mode}")
        df_prices = data.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Price")

        if px:
            fig = px.line(df_prices, x="Date", y="Price", color="Ticker",
                          title=f"Stock Prices Over Time ({suffix_label})",
                          labels={"Price": "Price (INR)", "Ticker": "Ticker"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(data)

        # Technicals (per-ticker)
        with st.expander("🧪 Technical Indicators (SMA 20/50/200, RSI 14)"):
            sel_for_ta = st.selectbox("Choose ticker for indicators", tickers_root)
            ta_series = data[sel_for_ta].dropna()
            indi = compute_indicators(ta_series)

            if px:
                fig_ta = go.Figure()
                fig_ta.add_trace(go.Scatter(x=indi.index, y=indi["Close"], mode="lines", name="Close"))
                fig_ta.add_trace(go.Scatter(x=indi.index, y=indi["SMA20"], mode="lines", name="SMA20"))
                fig_ta.add_trace(go.Scatter(x=indi.index, y=indi["SMA50"], mode="lines", name="SMA50"))
                fig_ta.add_trace(go.Scatter(x=indi.index, y=indi["SMA200"], mode="lines", name="SMA200"))
                fig_ta.update_layout(title=f"{sel_for_ta} — Price & SMAs", xaxis_title="Date", yaxis_title="Price (INR)")
                st.plotly_chart(fig_ta, use_container_width=True)

                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=indi.index, y=indi["RSI14"], mode="lines", name="RSI14"))
                fig_rsi.add_hline(y=70, line_dash="dot")
                fig_rsi.add_hline(y=30, line_dash="dot")
                fig_rsi.update_layout(title=f"{sel_for_ta} — RSI(14)", xaxis_title="Date", yaxis_title="RSI")
                st.plotly_chart(fig_rsi, use_container_width=True)
            else:
                st.line_chart(indi[["Close","SMA20","SMA50","SMA200"]])
                st.line_chart(indi[["RSI14"]])

        # ----------------- Normalized comparison -----------------
        st.subheader("📊 Normalized Stock Comparison (Relative Growth)")
        normalized = data / data.iloc[0] * 100
        df_norm = normalized.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Growth")
        if px:
            fig2 = px.line(df_norm, x="Date", y="Growth", color="Ticker",
                           title="Normalized Comparison (100 = first day)",
                           labels={"Growth": "Growth (Index)", "Ticker": "Ticker"})
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.line_chart(normalized)

        # ----------------- Performance table -----------------
        st.subheader("📈 Performance Metrics")
        freq = 252 if resample_mode == "Daily" else (52 if resample_mode == "Weekly" else 12)
        rows = []
        for t in data.columns:
            stats = perf_stats(data[t], freq_per_year=freq)
            if stats:
                rows.append({
                    "Ticker": t,
                    "Total Return": stats["Return"],
                    "CAGR": stats["CAGR"],
                    "Volatility": stats["Volatility"],
                    "Max Drawdown": stats["MaxDD"],
                    "Sharpe": stats["Sharpe"],
                })
        perf_df = pd.DataFrame(rows).set_index("Ticker")
        fmt = perf_df.applymap(lambda x: to_pct(x) if isinstance(x, (int,float,np.floating)) else x)
        st.dataframe(fmt)

        # ----------------- Correlation heatmap -----------------
        st.subheader("🔗 Correlation (Daily Returns)")
        rets = data.pct_change().dropna(how="all")
        corr = rets.corr().fillna(0)
        if px:
            figc = px.imshow(corr, text_auto=True, aspect="auto", title="Return Correlation Matrix")
            st.plotly_chart(figc, use_container_width=True)
        else:
            st.dataframe(corr)

        # ----------------- Portfolio backtest -----------------
        st.subheader("🧺 Portfolio Backtest")
        if not rets.empty:
            if weight_mode == "Equal Weight":
                w = np.array([1/len(data.columns)] * len(data.columns))
            else:
                # normalize custom weights; if all zero -> equal
                cw = np.array([custom_weights.get(t, 0) for t in data.columns], dtype=float)
                if cw.sum() == 0:
                    w = np.array([1/len(data.columns)] * len(data.columns))
                else:
                    w = cw / cw.sum()
            port_rets = (rets * w).sum(axis=1)
            port_curve = (1 + port_rets).cumprod()

            if px:
                figp = px.line(port_curve, title="Portfolio Cumulative Growth (Base = 1.0)")
                st.plotly_chart(figp, use_container_width=True)
            else:
                st.line_chart(port_curve)

            # Portfolio quick stats
            ps = perf_stats((port_curve * 100).rename("Px"), freq_per_year=freq)  # scale doesn't matter
            colA, colB, colC, colD = st.columns(4)
            colA.metric("Total Return", to_pct(ps["Return"]) if ps else "—")
            colB.metric("CAGR", to_pct(ps["CAGR"]) if ps else "—")
            colC.metric("Volatility", to_pct(ps["Volatility"]) if ps else "—")
            colD.metric("Sharpe", f"{ps['Sharpe']:.2f}" if ps and pd.notnull(ps["Sharpe"]) else "—")

            # Download portfolio series
            st.download_button(
                "⬇️ Download Portfolio Curve (CSV)",
                data=port_curve.rename("Portfolio").to_csv(index=True).encode("utf-8"),
                file_name=f"portfolio_curve_{suffix_label}.csv",
                mime="text/csv",
            )

        # ----------------- Fundamentals glance -----------------
        st.subheader("🏢 Fundamentals (Quick)")
        cols = st.columns(min(4, len(tickers_root)))
        for i, t in enumerate(tickers_root[:8]):  # show first up to 8 to keep UI tidy
            full_sym = add_suffix(t, suffix)
            fi = fast_fundamentals(full_sym)
            with cols[i % len(cols)]:
                st.markdown(f"**{t}**")
                st.caption(f"{full_sym}")
                if fi:
                    mc = fi.get("market_cap")
                    yrh = fi.get("year_high")
                    yrl = fi.get("year_low")
                    lp = fi.get("last_price")
                    cur = fi.get("currency","INR") or "INR"
                    st.write(f"Last: {lp if lp is not None else '—'} {cur}")
                    st.write(f"52w: {yrl if yrl is not None else '—'} — {yrh if yrh is not None else '—'}")
                    st.write(f"Market Cap: {f'{mc:,.0f}' if isinstance(mc,(int,float)) else '—'}")
                else:
                    st.write("—")

        # ----------------- Downloads -----------------
        st.subheader("⬇️ Downloads")
        st.download_button(
            label="Prices CSV",
            data=data.to_csv(index=True).encode("utf-8"),
            file_name=f"prices_{suffix_label}.csv",
            mime="text/csv",
        )
        st.download_button(
            label="Normalized CSV",
            data=normalized.to_csv(index=True).encode("utf-8"),
            file_name=f"normalized_{suffix_label}.csv",
            mime="text/csv",
        )

# -----------------------------
# Tips (collapsible)
# -----------------------------
with st.expander("ℹ️ Tips & Notes"):
    st.markdown(
        """
- Use **NSE (.NS)** for most Indian symbols (e.g., `RELIANCE.NS`, `TCS.NS`, `SBIN.NS`).  
- Use **BSE (.BO)** if you prefer BSE listings.  
- **All listed companies**: upload a CSV master list (column `symbol`) under *Import Full Symbol List*.  
  - You can export this from your internal database, a broker download, or a public CSV (raw GitHub).  
- Presets: **NIFTY 50** and **NIFTY BANK** buttons auto-fill common indices (via Yahoo Finance).  
- Indicators: open *Technical Indicators* to view SMA20/50/200 & RSI(14).  
- Portfolio: choose **Equal** or **Custom** weights; values auto-normalize if they don’t add to 100%.  
- Data source: Yahoo Finance; INR quotes for Indian tickers when available.  
"""
    )
