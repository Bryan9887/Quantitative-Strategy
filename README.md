# EMA + RSI Quant Strategy

A simple **quantitative trading strategy** for BTC/USDT using **EMA** and **RSI** indicators.

---

## Strategy

- **Indicators:** EMA50, EMA100, EMA200, RSI10  
- **Buy:** EMA100 > EMA200 & RSI < 30  
- **Sell:** RSI > 70 or EMA50 < EMA200  
- **Timeframe:** 1- Hour

---

## Features

- Fetch historical Binance data  
- Backtest trades & calculate profit, win rate  
- Visualize trades on candlestick charts with mplfinance  

---

## Usage

```bash
pip install pandas numpy ta mplfinance python-binance
python main.py
