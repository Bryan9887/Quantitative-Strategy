import pandas as pd
import mplfinance as mpf
import datetime
from binance.client import Client
import time
import ta  
import numpy as np


client = Client()  
#rsi + ema

#========================
#      Indicator 
#========================
def add_ema(df,window):
    df[f'ema{window}']=ta.trend.EMAIndicator(df["close"],window=window).ema_indicator()
    return df

def add_rsi(df,window):
    df[f'rsi{window}']=ta.momentum.RSIIndicator(df["close"],window=window).rsi()
    return df

def add_macd(df, fast=12, slow=26, signal=9):
    macd = ta.trend.MACD(
        close=df["close"],
        window_fast=fast,
        window_slow=slow,
        window_sign=signal
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    return df


def add_boll(df, window=20, ndev=2):
    boll = ta.volatility.BollingerBands(
        close=df["close"],
        window=window,
        window_dev=ndev
    )
    df["boll_mid"] = boll.bollinger_mavg()   # 中轨
    df["boll_upper"] = boll.bollinger_hband() # 上轨
    df["boll_lower"] = boll.bollinger_lband() # 下轨
    return df

#==========================


def ts_to_str(ts):
    """毫秒时间戳转成人类可读格式"""
    return datetime.datetime.fromtimestamp(ts/1000).strftime("%Y-%m-%d %H:%M:%S")


def get_history_klines(symbol, interval, start_str, end_str=None):
    df_all = []

    start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end_str).timestamp() * 1000) if end_str else None

    while True:
        print("请求中...", datetime.datetime.fromtimestamp(start_ts/1000))

        klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=start_ts,
            endTime=end_ts,
            limit=500  
        )

        if not klines:
            break

        df = pd.DataFrame(klines, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_asset_volume","number_of_trades",
            "taker_buy_base","taker_buy_quote","ignore"
        ])

        df_all.append(df)

        # 下一段开始时间 = 当前最后一根K线的 close_time
        last_close_time = klines[-1][6]
        start_ts = last_close_time + 1

        time.sleep(0.1)

        if end_ts and start_ts >= end_ts:
            break

    df = pd.concat(df_all, ignore_index=True)

    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)

    return df

#=================
#  Strategy 
#=================
def Strategy(df):
    #==============
    #  Initiallize 
    #==============
    df = add_ema(df,50)
    df = add_ema(df,200)
    df = add_ema(df,100)
    df = add_rsi(df,10)

    Buy_position = True
    buy_info = None
    sell_info = None
    win = 0
    #============
    #  Logic
    #============
    for i in range(len(df)):
        if i > 200:
            price = df.iloc[i]["close"]
            ema200 = df.iloc[i]["ema200"]
            ema50 = df.iloc[i]["ema50"]
            ema100 = df.iloc[i]["ema100"]
            rsi10 = df.iloc[i]["rsi10"]
            
            #==================
            # Buy Condition 
            #==================
            if Buy_position:
                
                if (ema100> ema200) and rsi10 <30:
                    Buy_position = False

                    buy_info = {
                        "index": i,
                        "time": ts_to_str(int(df.iloc[i]["open_time"])),
                        "price": float(price),
                        "ema50": float(ema50),
                        "ema100": float(ema100),
                        "ema200": float(ema200),
                        "rsi10": float(rsi10)
                    }

                    #print("\n=== BUY FOUND ===")
                    #print(buy_info)

            if not Buy_position:
                if (rsi10 >70) or ema50 <ema200:
                    Buy_position = True

                    sell_info = {
                        "index": i,
                        "time": ts_to_str(int(df.iloc[i]["open_time"])),
                        "price": float(price),
                        "ema50": float(ema50),
                        "ema200": float(ema200),
                        "ema100": float(ema100),
                        "rsi10": float(rsi10)
                    }    

                    #print("\n=== Sell FOUND ===")
                    #print(sell_info)
                    break

    # check trade or not
    if buy_info is None or sell_info is None:
        return pd.DataFrame(), pd.DataFrame(), None, None, False, 0, False



    #================
    #  Slicing
    #================
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('date', inplace=True)

    #plotting
    start = max(buy_info["index"] - 200, 0)
    end = min(sell_info["index"] + 50, len(df)-1)
    df_plot = df.iloc[start:end+1]

    buy_info["pos"] = buy_info["index"] - start
    sell_info["pos"] = sell_info["index"] - start

    #for next loop
    remain_start = sell_info["index"] + 1
    if remain_start >= len(df):
        df_remaining = pd.DataFrame()
    else:
        df_remaining = df.iloc[remain_start:]


    #profit calculation
    Profit = sell_info["price"] - buy_info["price"]
    Profit_percent = Profit / buy_info["price"] * 100
    print(f"Profit: {Profit:.2f}, Profit %: {Profit_percent:.2f}%")
    Trade = True

    if Profit_percent >0:
        win = True
    else:
        win = False

    return df_remaining, df_plot, buy_info, sell_info ,Trade, Profit_percent, win

#=====================
#   Graph Plotting
#=====================
def plot_graph(df,buy_info,sell_info):
    
    #buy_point = [None] * len(df_plot)
    #sell_point = [None] * len(df_plot)

    buy_point = np.full(len(df_plot), np.nan)
    sell_point = np.full(len(df_plot), np.nan)

    add=[]

    if (buy_info is not None) and (sell_info is not None):
        buy_point[buy_info["pos"]] = buy_info["price"]
        sell_point[sell_info["pos"]] = sell_info["price"]


        add.append(
            mpf.make_addplot(
                buy_point,
                type='scatter',
                marker='^',
                markersize = 50,
                color = 'green'
            )
        )

        add.append(
            mpf.make_addplot(
                sell_point,
                type='scatter',
                marker="v",
                markersize = 50,
                color = 'red'
            )
        )

        add.append(
            mpf.make_addplot(
                df['rsi10'],
                panel=2,
                color='blue',
                type='line',
                ylabel='RSI'
            )
        )

        mpf.plot(
            df,
            type='candle',
            style='charles',
            title='Chart',
            volume=True,
            mav=(100,200),
            figratio=(14,8),
            figscale=1.2,
            datetime_format='%Y-%m-%d',
            addplot=add
        )
#================
#   Initiallize 
#================
df = pd.DataFrame()
df_plot = pd.DataFrame()

buy_info = {}
sell_info = {}

count = -1  # -1 is offset because affer last trade will extra 1 count
Trade = True
Profit_percent = 0
Cummulative_profit = 0
win = True
win_count = 0
win_rate = 0
max_profit = 0
max_loss = 0

average_loss = 0
average_profit = 0

pure_loss = 0
pure_profit = 0

df = get_history_klines(
    symbol="BTCUSDT",
    interval=Client.KLINE_INTERVAL_15MINUTE,
    start_str="2025-01-01 00:00:00",
    end_str="2025-12-01 06:00:00"
)

while Trade:
    
    df, df_plot, buy_info, sell_info, Trade, Profit_percent, win = Strategy(df)

    #plot_graph(df_plot,buy_info,sell_info)

    
    count += 1
    Cummulative_profit += Profit_percent


    if win:
        win_count += 1
        pure_profit += Profit_percent
    else:
        pure_loss += Profit_percent
        

    #最大赚多少
    if Profit_percent >max_profit:
        max_profit = Profit_percent
    #最大回撤
    if Profit_percent < max_loss:
        max_loss = Profit_percent


win_rate = float(win_count/count) *100


print(f'Number of Trade: {count}')
print(f'Win Rate: {win_rate:.2f} %')
print(f'Largest Profit: {max_profit:.2f} %')
print(f'Largest Loss: {max_loss:.2f} %')
print(f'Average Profit: {(pure_profit/count):.2f} %')
print(f'Average Loss: {(pure_loss/count):.2f} %')
print(f'Final Profit: {Cummulative_profit:.2f} %') 
    
