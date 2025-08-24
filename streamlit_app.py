import streamlit as st
import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
import math
from io import StringIO

# Optional: Plotly for rich charts
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px, go = None, None

# =============================
# App Setup
# =============================
st.set_page_config(page_title="🚀 ShubhStocks: Smarter India Tracker", page_icon="💹", layout="wide")
st.title("🚀Smarter India Tracker ")
st.caption("Built by Shubh • 20 standout features")

# =============================
# Helpers & Data
# =============================

def add_suffix(symbol: str, sfx: str) -> str:
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol
    return f"{symbol}{sfx}"

@st.cache_data(show_spinner=False)
def fetch_history_close(tickers_full, start, end, interval="1d", auto_adjust=True):
    """Adjusted Close panel for multi-asset analytics."""
    frames = []
    for full in tickers_full:
        try:
            hist = yf.Ticker(full).history(start=start, end=end, interval=interval, auto_adjust=auto_adjust, actions=False)
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
def fetch_history_ohlc(ticker_full, start, end, interval="1d", auto_adjust=True):
    try:
        hist = yf.Ticker(ticker_full).history(start=start, end=end, interval=interval, auto_adjust=auto_adjust, actions=False)
        if not hist.empty:
            hist.index.name = "Date"
        return hist
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fast_fundamentals(full_symbol: str):
    try:
        t = yf.Ticker(full_symbol)
        fi = getattr(t, "fast_info", {})
        return {
            "currency": getattr(fi, "currency", None) if hasattr(fi, "currency") else (fi.get("currency") if isinstance(fi, dict) else None),
            "market_cap": getattr(fi, "market_cap", None) if hasattr(fi, "market_cap") else (fi.get("market_cap") if isinstance(fi, dict) else None),
            "year_high": getattr(fi, "year_high", None) if hasattr(fi, "year_high") else (fi.get("year_high") if isinstance(fi, dict) else None),
            "year_low": getattr(fi, "year_low", None) if hasattr(fi, "year_low") else (fi.get("year_low") if isinstance(fi, dict) else None),
            "last_price": getattr(fi, "last_price", None) if hasattr(fi, "last_price") else (fi.get("last_price") if isinstance(fi, dict) else None),
        }
    except Exception:
        return {}

# =============================
# Yahoo Finance Presets (stay Yahoo-only)
# =============================
@st.cache_data(show_spinner=True)
def load_yahoo_presets():
    presets = {"NIFTY50": [], "NIFTYBANK": []}
    try:
        presets["NIFTY50"] = [s.replace(".NS", "") for s in yf.tickers_nifty50()]
    except Exception:
        pass
    try:
        presets["NIFTYBANK"] = [s.replace(".NS", "") for s in yf.tickers_niftybank()]
    except Exception:
        pass
    return presets

PRESETS = load_yahoo_presets()

# =============================
# Sidebar — Universe & Dates
# =============================
st.sidebar.header("⭐ Universe & Controls")
if PRESETS.get("NIFTY50"):
    st.sidebar.success(f"NIFTY 50 loaded: {len(PRESETS['NIFTY50'])}")
if PRESETS.get("NIFTYBANK"):
    st.sidebar.info(f"NIFTY BANK loaded: {len(PRESETS['NIFTYBANK'])}")

exchange = st.sidebar.radio("Exchange", ["NSE (.NS)", "BSE (.BO)"], index=0)
suffix = ".NS" if "NSE" in exchange else ".BO"

preset_universe = sorted(set(PRESETS.get("NIFTY50", []) + PRESETS.get("NIFTYBANK", [])))
manual_add = st.sidebar.text_input("Add symbols (comma-separated, e.g., RELIANCE, TCS, SBIN)", value="")
file_up = st.sidebar.file_uploader("…or upload a CSV with a 'symbol' column", type=["csv"], key="user_csv")

user_syms = []
if file_up is not None:
    try:
        dfu = pd.read_csv(file_up)
        if "symbol" in dfu.columns:
            user_syms = [str(s).upper().strip() for s in dfu["symbol"].dropna().unique().tolist()]
    except Exception:
        pass
if manual_add.strip():
    user_syms.extend([s.strip().upper() for s in manual_add.split(",") if s.strip()])

ALL_COMPANIES = sorted(set(preset_universe + user_syms))

start_date = st.sidebar.date_input("Start Date", dt.date(2024, 1, 1))
end_date = st.sidebar.date_input("End Date", dt.date.today())
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
auto_adjust = st.sidebar.toggle("Use Adjusted Prices", value=True, help="Turn off to see raw OHLC without split/dividend adjustments")

selected_symbols = st.sidebar.multiselect("Select companies", ALL_COMPANIES[:800], ALL_COMPANIES[:6] if ALL_COMPANIES else [])

# =============================
# Guards
# =============================
if start_date > end_date:
    st.error("⚠️ Start date must be before end date.")
    st.stop()
if not selected_symbols:
    st.warning("Select at least one symbol from the sidebar.")
    st.stop()

full = [add_suffix(t, suffix) for t in selected_symbols]

# =============================
# Core Data Loads
# =============================
with st.spinner("Fetching price data from Yahoo Finance…"):
    prices = fetch_history_close(full, start_date, end_date, interval=interval, auto_adjust=auto_adjust)

if prices.empty:
    st.error("No data returned. Try different symbols/date range.")
    st.stop()

rets = prices.pct_change().dropna(how="all")

# =============================
# Tabs for 20 Differentiator Features
# =============================
T1, T2, T3, T4, T5 = st.tabs([
    "1) Price & Patterns",
    "2) Strategies & Alerts",
    "3) Risk & Portfolio",
    "4) Fundamentals & Events",
    "5) Options, Dividends & Reports",
])

# -------------------------------------------------
# 1) Price & Patterns (OHLC, Candles, Heatmaps, Network)
# -------------------------------------------------
with T1:
    st.subheader("1) Multi-asset Price View + Novel Visuals")
    if px:
        df_prices = prices.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Price")
        fig = px.line(df_prices, x="Date", y="Price", color="Ticker", title="Prices Over Time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(prices)

    st.markdown("**Candlestick + Auto Pattern Detection (per ticker)**")
    sel_ta = st.selectbox("Pick a ticker", selected_symbols, index=0, key="ta_ohlc")
    ohlc = fetch_history_ohlc(add_suffix(sel_ta, suffix), start_date, end_date, interval=interval, auto_adjust=auto_adjust)
    if not ohlc.empty and go is not None:
        candle = go.Figure(data=[go.Candlestick(x=ohlc.index, open=ohlc["Open"], high=ohlc["High"], low=ohlc["Low"], close=ohlc["Close"], name=sel_ta)])
        candle.update_layout(title=f"{sel_ta} — OHLC", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(candle, use_container_width=True)

        # Simple pattern: Bullish/Bearish Engulfing & Doji
        body = (ohlc["Close"] - ohlc["Open"]).abs()
        range_ = (ohlc["High"] - ohlc["Low"]).replace(0, np.nan)
        doji = (body / range_) < 0.1
        prev = ohlc.shift(1)
        bull_engulf = (ohlc["Close"] > ohlc["Open"]) & (prev["Close"] < prev["Open"]) & (ohlc["Close"] >= prev["Open"]) & (ohlc["Open"] <= prev["Close"])
        bear_engulf = (ohlc["Close"] < ohlc["Open"]) & (prev["Close"] > prev["Open"]) & (ohlc["Close"] <= prev["Open"]) & (ohlc["Open"] >= prev["Close"])
        last_marks = pd.DataFrame({"Doji": doji, "BullEngulf": bull_engulf, "BearEngulf": bear_engulf}).dropna(how="all").tail(10)
        st.dataframe(last_marks[last_marks.any(axis=1)].tail(10))

    st.markdown("**Correlation Heatmap (Returns)**")
    if px is not None:
        corr = rets.corr().fillna(0)
        st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto", title="Return Correlation Matrix"), use_container_width=True)
    else:
        st.dataframe(rets.corr())

    st.markdown("**Correlation Network (novel)**")
    thr = st.slider("Edge threshold (|corr|)", 0.0, 1.0, 0.6, 0.05)
    corr = rets.corr().fillna(0)
    nodes = corr.columns.tolist()
    # circular layout
    theta = np.linspace(0, 2*math.pi, len(nodes), endpoint=False)
    pos = {n: (math.cos(t), math.sin(t)) for n, t in zip(nodes, theta)}
    edges = [(i, j, corr.loc[i, j]) for i in nodes for j in nodes if i < j and abs(corr.loc[i, j]) >= thr]
    if go is not None and edges:
        edge_traces = []
        for i, j, w in edges:
            x0, y0 = pos[i]
            x1, y1 = pos[j]
            edge_traces.append(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="none", opacity=min(1, abs(w)), showlegend=False))
        node_trace = go.Scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text", text=nodes, textposition="top center")
        fig_net = go.Figure(data=edge_traces + [node_trace])
        fig_net.update_layout(title="Correlation Network", xaxis_visible=False, yaxis_visible=False)
        st.plotly_chart(fig_net, use_container_width=True)

# -------------------------------------------------
# 2) Strategies & Alerts (Backtests, Rules, Anomalies)
# -------------------------------------------------
with T2:
    st.subheader("2) Strategy Backtests + Smart Alerts")

    st.markdown("**A. Moving Average Crossover Backtest (per ticker)**")
    ma_ticker = st.selectbox("Ticker", selected_symbols, key="ma_ticker")
    short_win = st.number_input("Short MA", min_value=5, max_value=200, value=20, step=1)
    long_win = st.number_input("Long MA", min_value=10, max_value=400, value=50, step=1)
    p = prices[ma_ticker].dropna()
    if len(p) > long_win:
        sma_s = p.rolling(short_win).mean()
        sma_l = p.rolling(long_win).mean()
        signal = (sma_s > sma_l).astype(int)
        strat_rets = p.pct_change().fillna(0) * signal.shift(1).fillna(0)
        curve = (1 + strat_rets).cumprod()
        bench = (p / p.iloc[0])
        if px is not None:
            st.plotly_chart(px.line(pd.concat([curve.rename("Strategy"), bench.rename("Buy&Hold")], axis=1), title=f"{ma_ticker}: MA({short_win}/{long_win}) Backtest"), use_container_width=True)
        st.write({"Strategy CAGR": (curve.iloc[-1] ** (365.25 / max(1,(curve.index[-1]-curve.index[0]).days)) - 1) if len(curve)>1 else np.nan})

    st.markdown("**B. RSI Rules & Alerts (per ticker)**")
    rsi_ticker = st.selectbox("Ticker for RSI", selected_symbols, key="rsi_ticker")
    p2 = prices[rsi_ticker].dropna()
    if len(p2) > 15:
        delta = p2.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        overbought = rsi.iloc[-1] >= 70
        oversold = rsi.iloc[-1] <= 30
        st.metric("Latest RSI(14)", f"{rsi.iloc[-1]:.2f}", help="Alerts: OB>=70, OS<=30")
        st.info("Overbought 🚩" if overbought else ("Oversold ✅" if oversold else "Neutral"))
        if px is not None:
            st.plotly_chart(px.line(rsi, title=f"{rsi_ticker}: RSI(14)"), use_container_width=True)

    st.markdown("**C. Return Anomalies (Z-Score)**")
    roll = st.slider("Window (days)", 10, 120, 60)
    anomalies = {}
    for c in prices.columns:
        r = prices[c].pct_change()
        z = (r - r.rolling(roll).mean()) / (r.rolling(roll).std())
        if not z.empty and abs(z.iloc[-1]) >= 2:
            anomalies[c] = float(z.iloc[-1])
    if anomalies:
        st.warning(f"Anomaly candidates (|z|≥2): {anomalies}")
    else:
        st.caption("No significant anomalies today.")

# -------------------------------------------------
# 3) Risk & Portfolio (Efficient Frontier, VaR/CVaR, Beta, Monte Carlo)
# -------------------------------------------------
with T3:
    st.subheader("3) Portfolio Analytics & Risk Lab")

    freq = {"1d":252, "1wk":52, "1mo":12}[interval]
    mu = rets.mean() * freq
    sigma = rets.cov() * freq

    st.markdown("**A. Efficient Frontier (MPT)**")
    if len(prices.columns) >= 2 and go is not None:
        def port_stats(w):
            w = np.array(w)
            ret = float(np.dot(w, mu))
            vol = float(np.sqrt(np.dot(w, np.dot(sigma, w))))
            sr = ret/vol if vol>0 else np.nan
            return ret, vol, sr
        # random portfolios
        N = 2000
        Ws = np.random.dirichlet(np.ones(len(mu)), N)
        pts = np.array([port_stats(w) for w in Ws])
        figf = go.Figure()
        figf.add_trace(go.Scatter(x=pts[:,1], y=pts[:,0], mode="markers", name="Random", opacity=0.4))
        st.plotly_chart(figf.update_layout(title="Efficient Frontier (simulated)", xaxis_title="Volatility", yaxis_title="Return"), use_container_width=True)

    st.markdown("**B. One-click Tangency Portfolio (rf=0)**")
    if len(prices.columns) >= 2:
        try:
            inv = np.linalg.pinv(sigma.values)
            ones = np.ones(len(mu))
            w_tan = inv.dot(mu.values)
            w_tan = w_tan / w_tan.sum()
            w_series = pd.Series(w_tan, index=mu.index)
            st.dataframe((w_series*100).round(2).rename("% Weight"))
        except Exception:
            st.caption("Could not compute tangency weights.")

    st.markdown("**C. Risk: VaR & CVaR (parametric)**")
    alpha = st.slider("Confidence", 0.90, 0.99, 0.95, 0.01)
    port_w = np.array([1/len(prices.columns)]*len(prices.columns))
    port_rets = (rets * port_w).sum(axis=1)
    mu_d = port_rets.mean()
    sd_d = port_rets.std()
    from scipy.stats import norm
    var = -(mu_d + sd_d * norm.ppf(1-alpha))
    cvar = - (mu_d - sd_d * (norm.pdf(norm.ppf(1-alpha))/(1-alpha)))
    st.write({"Daily VaR": float(var), "Daily CVaR": float(cvar)})

    st.markdown("**D. Beta vs NIFTY 50 (Yahoo ^NSEI)**")
    try:
        idx = yf.Ticker("^NSEI").history(start=start_date, end=end_date, interval=interval)["Close"].pct_change().dropna()
        betas = {}
        for c in prices.columns:
            r = prices[c].pct_change().dropna()
            aligned = pd.concat([r, idx], axis=1).dropna()
            if len(aligned)>5:
                cov = np.cov(aligned.iloc[:,0], aligned.iloc[:,1])[0,1]
                betas[c] = float(cov / aligned.iloc[:,1].var()) if aligned.iloc[:,1].var()!=0 else np.nan
        st.dataframe(pd.Series(betas, name="Beta vs ^NSEI"))
    except Exception:
        st.caption("Index fetch failed; beta unavailable.")

    st.markdown("**E. Monte Carlo (GBM) — per ticker**")
    mc_t = st.selectbox("Ticker for MC", prices.columns, key="mc_t")
    sims = st.slider("Simulations", 100, 2000, 500, 100)
    horizon = st.slider("Days ahead", 30, 365, 180, 15)
    series = prices[mc_t].dropna()
    if len(series)>5:
        r = series.pct_change().dropna()
        mu_g = r.mean()
        sd_g = r.std()
        last = series.iloc[-1]
        rnd = np.random.normal(mu_g, sd_g, (horizon, sims))
        path = last * (1 + rnd).cumprod(axis=0)
        mean_path = path.mean(axis=1)
        if px is not None:
            st.plotly_chart(px.line(pd.DataFrame({"Mean": mean_path})), use_container_width=True)
        st.caption("Simple GBM-style simulation for indicative ranges.")

# -------------------------------------------------
# 4) Fundamentals & Events (Comparatives, Calendar, Sector Rotation)
# -------------------------------------------------
with T4:
    st.subheader("4) Fundamentals & Corporate Events")

    st.markdown("**A. Quick Fundamentals Glance**")
    cols = st.columns(min(4, len(full)))
    for i, root in enumerate(prices.columns[:8]):
        full_sym = add_suffix(root, suffix)
        fi = fast_fundamentals(full_sym)
        with cols[i % len(cols)]:
            st.markdown(f"**{root}**  ")
            st.caption(full_sym)
            st.write(f"Last: {fi.get('last_price', '—')} {fi.get('currency','INR') or 'INR'}")
            st.write(f"52w: {fi.get('year_low','—')} — {fi.get('year_high','—')}")
            mc = fi.get('market_cap')
            st.write(f"Mkt Cap: {f'{mc:,.0f}' if isinstance(mc,(int,float)) else '—'}")

    st.markdown("**B. Events Calendar (earnings, dividends, splits) — per ticker**")
    ev_t = st.selectbox("Ticker for events", prices.columns, key="events_t")
    T = yf.Ticker(add_suffix(ev_t, suffix))
    try:
        div = T.dividends
    except Exception:
        div = pd.Series(dtype=float)
    try:
        splits = T.splits
    except Exception:
        splits = pd.Series(dtype=float)
    try:
        cal = T.calendar if hasattr(T, 'calendar') else pd.DataFrame()
    except Exception:
        cal = pd.DataFrame()
    st.write("Dividends (recent):")
    st.dataframe(div.tail(10))
    st.write("Splits (recent):")
    st.dataframe(splits.tail(10))
    if isinstance(cal, pd.DataFrame) and not cal.empty:
        st.write("Upcoming/Recent earnings calendar (if available):")
        st.dataframe(cal)
    else:
        st.caption("No earnings calendar data available.")

    st.markdown("**C. Sector Rotation Snapshot (best effort)**")
    st.caption("Yahoo fast_info may not expose sectors consistently; this is a best-effort grouping if available via .info.")
    sectors = []
    for root in prices.columns:
        try:
            info = yf.Ticker(add_suffix(root, suffix)).info
            sectors.append({"Ticker": root, "Sector": info.get("sector", "Unknown")})
        except Exception:
            sectors.append({"Ticker": root, "Sector": "Unknown"})
    sec_df = pd.DataFrame(sectors)
    st.dataframe(sec_df)
    if not sec_df.empty:
        perf = (prices.iloc[-1] / prices.iloc[0] - 1).rename("Return")
        sec_perf = sec_df.set_index("Ticker").join(perf).groupby("Sector").mean().sort_values("Return", ascending=False)
        if px is not None:
            st.plotly_chart(px.bar(sec_perf, y="Return", title="Sector Rotation (avg return)").update_layout(yaxis_tickformat=",.0%"), use_container_width=True)
        else:
            st.dataframe(sec_perf)

# -------------------------------------------------
# 5) Options, Dividends & Export (Chain, Yield, Custom Index)
# -------------------------------------------------
with T5:
    st.subheader("5) Derivatives, Income & Reports")

    st.markdown("**A. Options Chain (if Yahoo provides it)**")
    opt_t = st.selectbox("Ticker for options", prices.columns, key="opt_t")
    try:
        tk = yf.Ticker(add_suffix(opt_t, suffix))
        exps = tk.options
        if exps:
            exp = st.selectbox("Expiry", exps, index=0)
            ch = tk.option_chain(exp)
            st.write("Calls:")
            st.dataframe(ch.calls.head(50))
            st.write("Puts:")
            st.dataframe(ch.puts.head(50))
        else:
            st.caption("No options data available for this symbol.")
    except Exception:
        st.caption("Options retrieval failed.")

    st.markdown("**B. Dividend Tracker & Yield**")
    div_t = st.selectbox("Ticker for dividends", prices.columns, key="div_t")
    try:
        dser = yf.Ticker(add_suffix(div_t, suffix)).dividends
        if isinstance(dser, pd.Series) and not dser.empty:
            last_year = dser[dser.index >= (dser.index.max() - pd.Timedelta(days=365))].sum()
            last_price = prices[div_t].iloc[-1]
            yield_est = float(last_year / last_price) if last_price and last_year else np.nan
            st.write({"12m Dividends": float(last_year), "Approx Yield": yield_est})
            if px is not None:
                st.plotly_chart(px.bar(dser.tail(20), title=f"{div_t}: Dividend History"), use_container_width=True)
        else:
            st.caption("No dividend data.")
    except Exception:
        st.caption("Dividend fetch failed.")

    st.markdown("**C. Custom Index Builder & Export**")
    st.caption("Create your own weighted index and backtest instantly.")
    weights = {}
    cols = st.columns(min(4, len(prices.columns)))
    for i, c in enumerate(prices.columns):
        with cols[i % len(cols)]:
            weights[c] = st.number_input(f"{c} %", min_value=0.0, max_value=100.0, value=round(100.0/len(prices.columns), 2))
    w = np.array([weights[c] for c in prices.columns])
    if w.sum() == 0:
        w = np.array([1/len(prices.columns)]*len(prices.columns))
    else:
        w = w / w.sum()
    port_curve = (1 + (rets * w).sum(axis=1)).cumprod()
    if px is not None:
        st.plotly_chart(px.line(port_curve, title="Custom Index (Base=1.0)"), use_container_width=True)
    st.download_button("⬇️ Download Custom Index CSV", data=port_curve.rename("CustomIndex").to_csv().encode("utf-8"), file_name="custom_index.csv", mime="text/csv")

# =============================
# Extras Section — Unique Add-ons
# =============================
with st.expander("📦 Extras: Save/Load Watchlist, Monthly Heatmap, Rolling Metrics"):
    st.markdown("**Save current selection as watchlist**")
    csv_buf = StringIO()
    pd.DataFrame({"symbol": selected_symbols}).to_csv(csv_buf, index=False)
    st.download_button("Save watchlist CSV", data=csv_buf.getvalue(), file_name="watchlist.csv")

    st.markdown("**Monthly Return Heatmap**")
    monthly = prices.resample("M").last().pct_change()
    if not monthly.empty:
        mh = monthly.copy()
        mh.index = mh.index.strftime("%Y-%m")
        if px is not None:
            st.plotly_chart(px.imshow(mh.T, aspect="auto", title="Monthly Returns", color_continuous_scale="RdBu_r"), use_container_width=True)
        else:
            st.dataframe(mh)

    st.markdown("**Rolling Sharpe & Max Drawdown (equal-weight portfolio)**")
    eq = (rets.mean(axis=1))
    roll = 63
    roll_ret = eq.rolling(roll).mean() * 252
    roll_vol = eq.rolling(roll).std() * np.sqrt(252)
    rsh = (roll_ret / roll_vol).replace([np.inf, -np.inf], np.nan)
    curve = (1 + eq).cumprod()
    dd = curve / curve.cummax() - 1
    if px is not None:
        st.plotly_chart(px.line(rsh, title="Rolling Sharpe (~3 months)"), use_container_width=True)
        st.plotly_chart(px.line(dd, title="Drawdown (Equal-weight)"), use_container_width=True)
    else:
        st.line_chart(rsh)
        st.line_chart(dd)

st.success("Loaded 20+ distinctive analytics and tools. All data sourced from Yahoo Finance via yfinance. Some fundamentals/events may be unavailable for certain tickers.")
