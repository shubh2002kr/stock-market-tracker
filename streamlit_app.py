# Indian Stock Market Tracker — Streamlit App
# All NSE companies + Beautiful UI + Smart Google Market News (always visible)
# Built by Shubh Kumar

import io
import re
from html import unescape
from datetime import datetime, timedelta, timezone
from io import StringIO
from urllib.parse import quote_plus
from email.utils import parsedate_to_datetime
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# -------------------------
# Page Config + Global Styles
# -------------------------
st.set_page_config(
    page_title="Indian Stock Market Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root{
  --bg: #0b1020;
  --card: rgba(255,255,255,0.06);
  --card-hover: rgba(255,255,255,0.10);
  --border: rgba(255,255,255,0.15);
  --text: #e9edf5;
  --muted: #a7b0c0;
  --brand1: #5b8cff;
  --brand2: #6be6b5;
  --accent: #ffd166;
}

html, body, [class^="stApp"]{
  background: radial-gradient(1200px 600px at 10% -10%, #18233d 10%, transparent 50%) no-repeat,
              radial-gradient(900px 500px at 110% -20%, #0d3a2d 5%, transparent 50%) no-repeat,
              linear-gradient(180deg, #0a0f1f 0%, #0b1020 100%) !important;
}

.app-hero{
  background: linear-gradient(135deg, rgba(91,140,255,.25), rgba(107,230,181,.15));
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 18px 22px;
  box-shadow: 0 10px 30px rgba(0,0,0,.25);
}
.app-title{ font-size: 1.9rem; font-weight: 800; color: var(--text); letter-spacing:.25px; }
.subtitle{ color: var(--muted); }

.card{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 14px 16px;
  transition: .2s ease;
  box-shadow: 0 8px 24px rgba(0,0,0,.15);
}
.card:hover{ background: var(--card-hover); transform: translateY(-2px); }

.metric-wrap .stMetric{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 8px 14px;
}

.pills .stButton>button{
  background: transparent; border:1px solid var(--border); color: var(--text);
  padding: 6px 12px; border-radius: 999px; cursor:pointer;
}
.pills .stButton>button:hover{ border-color: var(--brand2); background: rgba(107,230,181,.10); }

.news-grid{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
@media (max-width: 1000px){ .news-grid{ grid-template-columns: 1fr; } }
.news-card{
  background: linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,.03));
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 10px 14px;
  min-height: 120px;
}
.news-title a{ color: var(--text); text-decoration: none; font-weight: 700; }
.news-title a:hover{ text-decoration: underline; }
.news-meta{ color: var(--muted); font-size: .9rem; margin: .15rem 0 .35rem; }
.news-desc{ color: #e9edf5; opacity: .95; }

.footer {text-align:center; color: var(--muted); font-size: .9rem; padding-top: 10px;}
hr{ border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# 1-D Series Helper (prevents ndarray errors)
# -------------------------
def as_1d_float_series(x, index=None) -> pd.Series:
    arr = np.asarray(x)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if index is not None and len(index) == len(arr):
        return pd.Series(arr, index=index, dtype="float64")
    return pd.Series(arr, dtype="float64")

# -------------------------
# NSE Equity Master (robust fetch)
# -------------------------
NSE_SUFFIX = ".NS"
NSE_MASTER_URLS = [
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
    "https://www.nseindia.com/content/equities/EQUITY_L.csv",
]
NSE_WARMUP_URLS = [
    "https://www.nseindia.com/",
    "https://www.nseindia.com/market-data/securities-available-for-trading",
]
BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Connection": "keep-alive",
}
CSV_HEADERS = {**BROWSER_HEADERS, "Accept": "text/csv,*/*;q=0.8",
               "Referer": "https://www.nseindia.com/market-data/securities-available-for-trading"}

@st.cache_data(show_spinner=False)
def fetch_nse_equity_master() -> pd.DataFrame:
    last_err = None
    with requests.Session() as sess:
        for wurl in NSE_WARMUP_URLS:
            try: sess.get(wurl, headers=BROWSER_HEADERS, timeout=10)
            except Exception as e: last_err = e
        for url in NSE_MASTER_URLS:
            try:
                r = sess.get(url, headers=CSV_HEADERS, timeout=15)
                r.raise_for_status()
                text = r.text if isinstance(r.text, str) else r.content.decode("utf-8", errors="ignore")
                df = pd.read_csv(StringIO(text))
                df = df.rename(columns={c: c.strip() for c in df.columns})
                df = df[df.get("SYMBOL").notna() & (df["SYMBOL"].astype(str).str.strip() != "")]
                df["yahoo"] = df["SYMBOL"].astype(str).str.strip() + NSE_SUFFIX
                name_col = next((c for c in ["NAME OF COMPANY","NAME_OF_COMPANY","NAMEOF COMPANY","NAMEOF_COMPANY"] if c in df.columns), None)
                disp_name = df[name_col] if name_col else df["SYMBOL"]
                df["label"] = disp_name.astype(str).str.strip() + " (" + df["yahoo"] + ")"
                keep = ["label","yahoo","SYMBOL"] + [c for c in ["SERIES","ISIN"] if c in df.columns]
                return df[keep].sort_values("label").reset_index(drop=True)
            except Exception as e:
                last_err = e
                continue
    # Return empty df instead of raising -> so news can still show
    return pd.DataFrame(columns=["label","yahoo","SYMBOL"])

# -------------------------
# Price / Info / Indicators
# -------------------------
@st.cache_data(show_spinner=False)
def fetch_prices(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if not df.empty:
        df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"})
    return df

@st.cache_data(show_spinner=False)
def info_for(ticker: str) -> dict:
    try: return yf.Ticker(ticker).info or {}
    except Exception: return {}

def sma(series: pd.Series, window: int) -> pd.Series:
    s = as_1d_float_series(series, getattr(series, "index", None))
    return s.rolling(window).mean()

def ema(series: pd.Series, window: int) -> pd.Series:
    s = as_1d_float_series(series, getattr(series, "index", None))
    return s.ewm(span=window, adjust=False).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    s = as_1d_float_series(series, getattr(series, "index", None))
    delta = s.diff()
    up = delta.where(delta > 0, 0.0)
    down = (-delta).where(delta < 0, 0.0)
    roll_up = up.rolling(window=window, min_periods=window).mean()
    roll_down = down.rolling(window=window, min_periods=window).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))

# -------------------------
# Google News (Market-wide)
# -------------------------
GOOGLE_NEWS_BASE = "https://news.google.com/rss/search"

def _strip_html(html_text: str) -> str:
    if not html_text: return ""
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = re.sub(r"\s+", " ", text).strip()
    return unescape(text)

@st.cache_data(show_spinner=False)
def fetch_google_news(query: str, days: int = 7, lang_region: str = "en-IN", ceid: str = "IN:en", max_items: int = 40) -> pd.DataFrame:
    q = f"{query} when:{days}d"
    url = f"{GOOGLE_NEWS_BASE}?q={quote_plus(q)}&hl={lang_region}&gl=IN&ceid={ceid}"
    headers = {
        "User-Agent": BROWSER_HEADERS["User-Agent"],
        "Accept": "application/rss+xml,text/xml;q=0.9,*/*;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    items = []
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_raw = item.findtext("pubDate")
        try:
            pub_dt = parsedate_to_datetime(pub_raw) if pub_raw else None
            if pub_dt and pub_dt.tzinfo is None: pub_dt = pub_dt.replace(tzinfo=timezone.utc)
        except Exception: pub_dt = None
        source = item.findtext("source") or ""
        desc = item.findtext("description") or ""
        snippet = _strip_html(desc)
        items.append({
            "title": title, "link": link, "provider": source.strip(),
            "published": pub_dt, "published_str": pub_dt.strftime("%Y-%m-%d %H:%M UTC") if pub_dt else "",
            "snippet": snippet,
        })
        if len(items) >= max_items: break
    df = pd.DataFrame(items)
    if not df.empty and "published" in df.columns:
        df = df.sort_values("published", ascending=False, na_position="last")
    return df.reset_index(drop=True)

# -------------------------
# Header
# -------------------------
hero = st.container()
with hero:
    st.markdown(
        '<div class="app-hero">'
        '<div class="app-title">📈 Indian Stock Market Tracker</div>'
        '<div class="subtitle">Built by <b>Shubh Kumar</b> · All NSE companies · Live charts · Market news</div>'
        '</div>',
        unsafe_allow_html=True
    )

st.write("")  # spacing

# -------------------------
# Sidebar Controls (styled)
# -------------------------
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    period = st.selectbox("Period",
                          options=["1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"],
                          index=3, help="How far back to load data")
    end_date = datetime.now()
    if period == "max":
        start_date = datetime(1990,1,1)
    elif period == "ytd":
        start_date = datetime(end_date.year,1,1)
    else:
        qty = int("".join([c for c in period if c.isdigit()]))
        unit = "".join([c for c in period if c.isalpha()])
        start_date = end_date - timedelta(days=(30*qty if unit=="mo" else 365*qty))
    st.markdown("---")
    show_ma = st.checkbox("SMA 20/50 + EMA 20", value=True)
    show_rsi = st.checkbox("RSI (14)", value=True)
    st.markdown("---")
    st.caption("If NSE blocks the live CSV, upload EQUITY_L.csv below.")
    _ = st.file_uploader("Upload EQUITY_L.csv (fallback)", type=["csv"])

# -------------------------
# Load NSE list (never stops the app)
# -------------------------
with st.spinner("Loading all NSE-listed companies…"):
    nse_list = fetch_nse_equity_master()
    if nse_list.empty:
        st.warning("Could not load the NSE company list right now. Charts may be unavailable, but Market News is shown below.")

# -------------------------
# Company Selector (card)
# -------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
left, right = st.columns([0.7, 0.3])
with left:
    pick = st.selectbox("🔎 Search company (NSE)",
                        options=nse_list["label"].tolist() if not nse_list.empty else [],
                        index=None,
                        placeholder="Type to search all NSE companies…")
with right:
    compare = st.toggle("Compare Mode", value=False)
st.markdown('</div>', unsafe_allow_html=True)

if not compare:
    symbols = [nse_list.loc[nse_list["label"] == pick, "yahoo"].iloc[0]] if (pick and not nse_list.empty) else []
else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    picks = st.multiselect("➕ Add companies to compare (max 5)",
                           options=nse_list["label"].tolist() if not nse_list.empty else [],
                           default=[], max_selections=5,
                           placeholder="Type to search and add…")
    st.markdown('</div>', unsafe_allow_html=True)
    symbols = nse_list.loc[nse_list["label"].isin(picks), "yahoo"].tolist() if not nse_list.empty else []

# -------------------------
# Charts / Metrics FIRST (if we have symbols)
# -------------------------
if symbols:
    tabs = st.tabs(symbols)
    for i, sym in enumerate(symbols):
        with tabs[i]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"### {sym}")

            with st.spinner("Fetching data…"):
                df = fetch_prices(sym, start=start_date, end=end_date + timedelta(days=1))
                info = info_for(sym)

            if df.empty:
                st.warning("No data found for this ticker. Try another symbol.")
                st.markdown('</div>', unsafe_allow_html=True)
                continue

            # 1-D series
            s_open = as_1d_float_series(df["open"], df.index)
            s_high = as_1d_float_series(df["high"], df.index)
            s_low  = as_1d_float_series(df["low"],  df.index)
            s_close = as_1d_float_series(df["close"], df.index)
            s_vol = as_1d_float_series(df.get("volume", []), df.index) if "volume" in df.columns else pd.Series(dtype="float64")

            # Metrics
            st.markdown('<div class="metric-wrap">', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            last_close = float(s_close.iloc[-1])
            prev_close = float(s_close.iloc[-2]) if len(s_close) > 1 else last_close
            pct = ((last_close - prev_close) / prev_close) * 100 if prev_close else 0.0
            try: high_52w = float(s_close.rolling(252).max().iloc[-1])
            except Exception: high_52w = float("nan")
            try: low_52w = float(s_close.rolling(252).min().iloc[-1])
            except Exception: low_52w = float("nan")
            last_vol_val = int(s_vol.iloc[-1]) if len(s_vol) and pd.notna(s_vol.iloc[-1]) else 0
            m1.metric("Last Close", f"{last_close:,.2f}", f"{pct:+.2f}%")
            m2.metric("52W High", f"{high_52w:,.2f}" if pd.notna(high_52w) else "—")
            m3.metric("52W Low", f"{low_52w:,.2f}" if pd.notna(low_52w) else "—")
            m4.metric("Volume (last)", f"{last_vol_val:,}")
            st.markdown('</div>', unsafe_allow_html=True)

            # Price chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=s_open, high=s_high, low=s_low, close=s_close, name="OHLC"))
            if show_ma:
                fig.add_trace(go.Scatter(x=df.index, y=s_close.rolling(20).mean(), name="SMA 20"))
                fig.add_trace(go.Scatter(x=df.index, y=s_close.rolling(50).mean(), name="SMA 50"))
                fig.add_trace(go.Scatter(x=df.index, y=s_close.ewm(span=20, adjust=False).mean(), name="EMA 20"))
            fig.update_layout(
                height=520, margin=dict(l=10,r=10,t=30,b=10),
                xaxis_title="Date", yaxis_title="Price",
                legend=dict(orientation="h"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)

            # RSI chart
            if show_rsi:
                r = rsi(s_close)
                rfig = go.Figure()
                rfig.add_trace(go.Scatter(x=df.index, y=r, name="RSI 14"))
                rfig.add_hline(y=70, line_dash="dot")
                rfig.add_hline(y=30, line_dash="dot")
                rfig.update_layout(
                    height=250, margin=dict(l=10,r=10,t=10,b=10),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(rfig, use_container_width=True)

            # Company info
            with st.expander("Company Overview"):
                c1,c2,c3 = st.columns(3)
                c1.markdown(f"**Name:** {info.get('longName') or info.get('shortName') or '—'}")
                c2.markdown(f"**Sector:** {info.get('sector','—')}")
                c3.markdown(f"**Industry:** {info.get('industry','—')}")
                c1,c2,c3 = st.columns(3)
                mc = info.get("marketCap")
                c1.markdown(f"**Market Cap:** {mc:,.0f}" if mc else "**Market Cap:** —")
                c2.markdown(f"**PE (TTM):** {info.get('trailingPE','—')}")
                c3.markdown(f"**Dividend Yield:** {info.get('dividendYield','—')}")
                st.caption("Data source: Yahoo Finance (some fields may be missing).")

            # Export
            csv_buf = io.StringIO()
            df.to_csv(csv_buf)
            st.download_button("⬇️ Download price data (CSV)", data=csv_buf.getvalue(),
                               file_name=f"{sym.replace('.', '_')}_prices.csv", mime="text/csv",
                               use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Market News (Google) — ALWAYS VISIBLE
# (renders regardless of symbols or NSE list availability)
# Auto-includes selected company names with a toggle
# -------------------------
base_market_query = "Indian stock market OR Sensex OR Nifty OR NSE OR BSE"

selected_labels = []
if symbols:
    # if nse_list loaded, map yahoo→label; else fallback to symbols themselves
    try:
        selected_labels = nse_list.loc[nse_list["yahoo"].isin(symbols), "label"].tolist()
    except Exception:
        selected_labels = symbols[:]
selected_names = [lbl.split(" (")[0].strip() for lbl in selected_labels][:3]

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 🗞️ Market News (Google)")

c1, c2 = st.columns([0.75, 0.25])
with c1:
    auto_include = st.toggle("Include selected companies in news", value=True,
                             help="When ON, the Google query includes the companies you picked above.")
with c2:
    google_days = st.slider("Lookback (days)", 1, 30, 7)

if auto_include and selected_names:
    company_query = " OR ".join([f'"{name}"' for name in selected_names])
    smart_default_query = f"({company_query}) OR ({base_market_query})"
else:
    smart_default_query = base_market_query

google_query = st.text_input("News query", value=smart_default_query,
                             help='Tip: You can edit this. Use OR, quotes, e.g., "Infosys" OR "TCS"')

with st.spinner("Fetching Google News…"):
    try:
        gdf = fetch_google_news(google_query, days=google_days, lang_region="en-IN", ceid="IN:en", max_items=40)
    except Exception as e:
        st.error(f"Could not fetch Google News right now. {e}")
        gdf = pd.DataFrame()

if gdf.empty:
    st.info("No Google News items found.")
else:
    st.markdown('<div class="news-grid">', unsafe_allow_html=True)
    for _, row in gdf.iterrows():
        ttl = row["title"]; lnk = row["link"]
        meta = " · ".join([x for x in [row.get("provider",""), row.get("published_str","")] if x])
        snip = row.get("snippet", "")
        st.markdown(
            f'<div class="news-card">'
            f'<div class="news-title"><a href="{lnk}" target="_blank">{ttl}</a></div>'
            f'<div class="news-meta">{meta}</div>'
            f'<div class="news-desc">{snip}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Footer
# -------------------------
st.markdown('<hr/>', unsafe_allow_html=True)
st.markdown(f'<div class="footer">© {datetime.now().year} · Indian Stock Market Tracker · Built by Shubh Kumar</div>', unsafe_allow_html=True)
