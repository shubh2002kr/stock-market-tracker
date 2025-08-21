import streamlit as st
import datetime
import pandas as pd
import yfinance as yf

# Try Plotly; fall back to Streamlit charts if not installed
try:
    import plotly.express as px
except Exception:
    px = None

# -----------------------------
# Streamlit App Title with Unique Branding
# -----------------------------
st.set_page_config(page_title="📈 S.H.U.B.H. Stock Tracker", page_icon="📊", layout="wide")

st.title("📈 S.H.U.B.H. Stock Tracker")
st.caption("Smart Hub for Understanding Business Holdings")
st.write("Easily track stock prices and compare multiple companies using Yahoo Finance data.")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("📊 Stock Selection")

# 25 popular tickers across sectors
COMPANY_LIST = [
    "AAPL", "GOOG", "MSFT", "TSLA", "AMZN", "META", "NVDA", "NFLX",
    "INTC", "AMD", "IBM", "ORCL", "CSCO", "ADBE", "PYPL", "UBER",
    "SHOP", "BABA", "V", "MA", "JPM", "BAC", "WMT", "T", "PFE"
]

DEFAULT_TICKERS = ["AAPL", "GOOG", "MSFT", "TSLA"]

tickers = st.sidebar.multiselect(
    "Select Stocks to Compare",
    options=COMPANY_LIST,
    default=DEFAULT_TICKERS,
)

# Date range selection
start_date = st.sidebar.date_input("Start Date", datetime.date(2024, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# -----------------------------
# Fetch Data Function (robust & cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(selected, start, end):
    """Download adjusted close prices per ticker and combine into one DataFrame.
    This avoids MultiIndex pitfalls of yf.download with multiple tickers.
    """
    frames = []
    for t in selected:
        try:
            hist = yf.Ticker(t).history(start=start, end=end, auto_adjust=True, actions=False)
            if not hist.empty and "Close" in hist.columns:
                frames.append(hist["Close"].rename(t))
        except Exception:
            # Ignore individual ticker failures to keep app running
            continue
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, axis=1).sort_index()
    data.index.name = "Date"
    return data

# -----------------------------
# Main App Logic
# -----------------------------
if start_date > end_date:
    st.error("⚠️ Start date must be before end date.")
elif not tickers:
    st.warning("Please select at least one stock to display data.")
else:
    with st.spinner("Fetching data..."):
        data = load_data(tickers, start_date, end_date)

    if data.empty:
        st.error("No data returned. Try different tickers or a different date range.")
    else:
        # Raw data preview
        if st.checkbox("Show Raw Data"):
            st.dataframe(data.tail())

        # Price chart
        df_prices = data.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Price")
        if px:
            fig = px.line(
                df_prices,
                x="Date",
                y="Price",
                color="Ticker",
                title="Stock Prices Over Time",
                labels={"Price": "Price (USD)", "Ticker": "Ticker"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Plotly is not installed; showing a basic chart. Install with: pip install plotly")
            st.line_chart(data)

        # Normalized comparison
        st.subheader("📊 Normalized Stock Comparison (Relative Growth)")
        normalized = data / data.iloc[0] * 100
        df_norm = normalized.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Growth")
        if px:
            fig2 = px.line(
                df_norm,
                x="Date",
                y="Growth",
                color="Ticker",
                title="Normalized Stock Comparison (Relative Growth)",
                labels={"Growth": "Growth (%)", "Ticker": "Ticker"},
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.line_chart(normalized)

        # Download buttons
        st.download_button(
            label="⬇️ Download Prices CSV",
            data=data.to_csv(index=True).encode("utf-8"),
            file_name="prices.csv",
            mime="text/csv",
        )
        st.download_button(
            label="⬇️ Download Normalized CSV",
            data=normalized.to_csv(index=True).encode("utf-8"),
            file_name="normalized_prices.csv",
            mime="text/csv",
        )
