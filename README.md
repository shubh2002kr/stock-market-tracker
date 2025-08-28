# 📈 Indian Stock Market Tracker — Streamlit App  
> **Built by Shubh Kumar**  

[![Streamlit](https://img.shields.io/badge/Made%20With-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)  
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)  
[![Plotly](https://img.shields.io/badge/Charts-Plotly-3DDC84?logo=plotly&logoColor=white)](https://plotly.com/python/)  

An **interactive dashboard** to track **Indian stock markets (NSE-listed companies)** with **live prices, charts, technical indicators, and Google Market News** — all in one place with a sleek UI.  

---

## ✨ Features
- 🔎 **Search & Compare Companies** → All NSE-listed companies, up to 5 side-by-side  
- 📊 **Charts & Indicators** →  
  - Candlestick OHLC chart  
  - SMA (20, 50) & EMA (20)  
  - RSI (14) with overbought/oversold levels  
- 📈 **Key Metrics** → Last Close, % Change, 52W High/Low, Volume  
- 🏢 **Company Info** → Market Cap, PE Ratio, Dividend Yield, Sector/Industry  
- 🗞️ **Google Market News** → Always visible, auto-includes selected companies  
- ⬇️ **Download CSV** → Export historical price data instantly  
- 🎨 **Beautiful UI** → Dark theme, modern cards, responsive layout  

---

## 📸 Screenshots  

👉 *(Add your screenshots here to make the repo stand out)*  

| Dashboard Home | Stock Charts | Market News |
|---------------|--------------|-------------|
| ![Dashboard](https://via.placeholder.com/400x220.png?text=Dashboard+Home) | ![Charts](https://via.placeholder.com/400x220.png?text=Stock+Charts) | ![News](https://via.placeholder.com/400x220.png?text=Market+News) |

---

## ⚡ Quickstart
```bash
# Clone repo
git clone https://github.com/<your-username>/indian-stock-market-tracker.git
cd indian-stock-market-tracker

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
