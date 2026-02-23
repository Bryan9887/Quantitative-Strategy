import pandas as pd
import mplfinance as mpf
import datetime
from binance.client import Client
import time
import ta  
import numpy as np
import matplotlib.pyplot as plt
import random
import keyboard
import matplotlib.pyplot as plt

try:
    client = Client()
except Exception as e:
    print("Binance API unreachable:", e)


class Bitcoin():
    def __init__(self):
        self.df = pd.DataFrame()  # 保存 K 线数据
        
        self.swing_highs = []
        self.swing_lows = []

        self.big_highs = []
        self.big_lows = []

        self.low_gradient = []
        self.high_gradient = []
        self.low_trend_score = []
        self.high_trend_score = []

        self.current_trend_score = 0
        self.current_trend_list = []

        self.latest_high = {}
        self.latest_low = {}
        self.rsi = "rsi14"


        self.price_range = 0
        self.macd_range = 0
        self.rsi_range = 0

        self.trend_score = 0
        self.bg_trend_score = 0
        self.Trend_decision = 0

        self.trend_inverse = {
            "bool": False,
            "time": None,
            "unix": None,
            "from": None,
            "to": None
        }
        self.prev_trend_name = None
        self.trend_name = None

        # ===== Trade state (future) =====
        self.position_state = 0
        self.entry = []
        self.exit = []

    #==============
    # Indicator 
    #==============
    # ================== 指标函数 ==================
    def add_ema(self, window):
        self.df[f'ema{window}'] = ta.trend.EMAIndicator(self.df["close"], window=window).ema_indicator()

    def add_rsi(self, window):
        self.df[f'rsi{window}'] = ta.momentum.RSIIndicator(self.df["close"], window=window).rsi()

    def add_macd(self, fast=12, slow=26, signal=9):
        macd = ta.trend.MACD(self.df['close'], window_slow=slow, window_fast=fast, window_sign=signal)
        self.df['macd'] = macd.macd()
        self.df['macd_signal'] = macd.macd_signal()
        self.df['macd_hist'] = macd.macd_diff()

    # ================== 获取随机 K 线 ==================
    def random_klines(self, symbol, interval, start_str, end_str=None):
        start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_str).timestamp() * 1000) if end_str else None
        random_start_ts = random.randint(start_ts, end_ts)

        klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=random_start_ts,
            endTime=end_ts,
            limit=220
        )
        if not klines:
            print("没有获取到数据")
            return

        self.df = pd.DataFrame(klines, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_asset_volume","number_of_trades",
            "taker_buy_base","taker_buy_quote","ignore"
        ])
        for col in ["open","high","low","close","volume"]:
            self.df[col] = self.df[col].astype(float)
        self.df["open_time"] = pd.to_datetime(self.df["open_time"], unit="ms")
        self.df.set_index("open_time", inplace=True)

    #==================================================================
    def get_history_klines(self,symbol, interval, start_str, end_str=None):

        df_all = []

        start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_str).timestamp() * 1000) if end_str else None

        while True:
            #print("请求中...", datetime.datetime.fromtimestamp(start_ts/1000))

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

            #time.sleep(0.1)

            if end_ts and start_ts >= end_ts:
                break

        df = pd.concat(df_all, ignore_index=True)

        for col in ["open","high","low","close","volume"]:
            df[col] = df[col].astype(float)

        return df
    

    # ================== 增量拉取 50 根 ==================
    def add_klines(self, symbol, interval, n):
        if self.df.empty:
            print("self.df 为空，先用 random_klines 初始化")
            return
        last_ts = int(self.df.index[-1].timestamp() * 1000)
        end_ts = last_ts + n * 15 * 60 * 1000  # 5 分钟间隔
        klines = client.get_klines(
            symbol=symbol, 
            interval=interval,
            startTime=last_ts + 1,
            endTime=end_ts,
            limit=n
        )
        if not klines:
            print("没有获取到新数据")
            return
        
        # 只用原始的 12 列
        cols = ["open_time","open","high","low","close","volume",
                "close_time","quote_asset_volume","number_of_trades",
                "taker_buy_base","taker_buy_quote","ignore"]
        df_new = pd.DataFrame(klines, columns=cols)
        
        for col in ["open","high","low","close","volume"]:
            df_new[col] = df_new[col].astype(float)
        df_new["open_time"] = pd.to_datetime(df_new["open_time"], unit="ms")
        df_new.set_index("open_time", inplace=True)
        
        # 合并
        self.df = pd.concat([self.df, df_new])
        self.df = self.df[~self.df.index.duplicated(keep='last')]

        self.df = self.df.iloc[n:]

    #=================================================

    def fix_duplicate_price(self, col="close", max_gap=10, offset=0.1):
        """
        如果 price 相同 且 index 距离 < max_gap
        后面的那一根 price += offset
        """

        for i in range(len(self.df)):
            price_i = self.df[col].iloc[i]

            # 只往前看 max_gap 根，效率高
            start = max(0, i - max_gap)

            for j in range(start, i):
                if self.df[col].iloc[j] == price_i:
                    # ✅ 改“后面”的
                    self.df.iloc[i, self.df.columns.get_loc(col)] += offset
                    break

    #=============================================================


    # ================== 按 Space 增量 ==================
    def live_update(self, symbol, interval,numbers):
        print(f'增量获取 {numbers} 根 K 线...')

        self.add_klines(symbol, interval,numbers)
        self.fix_duplicate_price(col="close", max_gap=10, offset=0.1)
        self.add_ema(8)
        self.add_ema(26)
        self.add_ema(200)
        self.add_rsi(14)
        self.add_macd()
        self.Strategy()
        self.plot_graph()
        print("更新完成！")
        time.sleep(0.5)  # 防止连续触发

    #=============================================
    #======================
    #    Swing
    # =====================
    def is_swing_high(self, i, n, high_col):


        if i < n or i + n >= len(self.df):
            return None

        current = self.df[high_col].iloc[i]
        left_side  = self.df[high_col].iloc[i-n:i]
        right_side = self.df[high_col].iloc[i+1:i+1+int(n*0.6)]

        time_unix = int(self.df.index[i].timestamp())


        if (current >= left_side.max()) and (current >= right_side.max()):

            rsi_window  = self.df[self.rsi].iloc[i-5:i+6]
            macd_window = self.df['macd_signal'].iloc[i-5:i+6].dropna()

            rsi_id = rsi_window.idxmax()

            if macd_window.empty:
                macd = np.nan
            else:
                macd_id = macd_window.idxmax()
                macd = macd_window.loc[macd_id]
                #print(self.df.index[i])
                #print(macd_id)

            return {
                "time": self.df.index[i],      # ⭐ 时间（不会走位）
                "unix": time_unix,
                "price": current,
                "index": i,
                "macd": macd,
                "rsi": self.df[self.rsi].loc[rsi_id]
            }

        return None
    
    def is_swing_low(self, i, n, high_col):

        if i < n or i + n >= len(self.df):
            return None

        current = self.df[high_col].iloc[i]

        left_side  = self.df[high_col].iloc[i-n:i]
        right_side = self.df[high_col].iloc[i+1:i+1+int(n*0.6)]

        time_unix = int(self.df.index[i].timestamp())


        if (current <= left_side.min()) and (current <= right_side.min()):
            rsi_window  = self.df[self.rsi].iloc[i-5:i+6]
            macd_window = self.df['macd_signal'].iloc[i-5:i+6].dropna()

            rsi_id = rsi_window.idxmin()

            if macd_window.empty:
                macd = np.nan
            else:
                macd_id = macd_window.idxmin()
                macd = macd_window.loc[macd_id]

            return {
                "time": self.df.index[i],      # ⭐ 时间（不会走位）
                "price": current,
                "unix": time_unix,
                "index": i,
                "macd": macd,
                "rsi": self.df[self.rsi].loc[rsi_id]
            }
        

        return None
    
    def find_latest_swing_high(self, n, col):
        """
        从最新K线往回找最近的一个 swing high
        """
        for i in range(len(self.df) - 1, -1, -1):
            h = self.is_swing_high(i, n, col)
            if h:
                return h
        return None

    def find_latest_swing_low(self, n, col):
        """
        从最新K线往回找最近的一个 swing low
        """
        for i in range(len(self.df) - 1, -1, -1):
            l = self.is_swing_low(i, n, col)
            if l:
                return l
        return None

    def unique_swing_highs(self, start_gap, max_gap, col,num_high,length):
        """
        自动增加 gap，直到整张图里只剩 2 个 swing high
        """
        true_length = min(length,len(self.df))
        starting_point = len(self.df) - true_length
        for n in range(start_gap, max_gap + 1):
            highs = []

            for i in range(starting_point,len(self.df)):
                h = self.is_swing_high(i, n, col)
                if h:
                    highs.append(h)

            if  1 <= len(highs) <= num_high:
                #print(highs)
                return highs


        return []

    def unique_swing_lows(self, start_gap, max_gap, col,num_low,length):
        """
        自动增加 gap，直到整张图里只剩 2 个 swing high
        """
        true_length = min(length,len(self.df))
        starting_point = len(self.df) - true_length

        for n in range(start_gap, max_gap + 1):
            lows = []

            for i in range(starting_point,len(self.df)):
                h = self.is_swing_low(i, n, col)
                if h:
                    lows.append(h)

            if  1 <= len(lows) <= num_low:
                return lows

        return []

    def fixed_swing_highs_20(self, col, length=None):
        """
        扫描固定 gap=20 的 swing highs
        """
        GAP = 25
        highs = []

        true_length = len(self.df) if length is None else min(length, len(self.df))
        start = len(self.df) - true_length

        for i in range(start, len(self.df)):
            h = self.is_swing_high(i, GAP, col)
            if h:
                highs.append(h)

        return highs


    def fixed_swing_lows_20(self, col, length=None):
        """
        扫描固定 gap=20 的 swing lows
        """
        GAP = 25
        lows = []

        true_length = len(self.df) if length is None else min(length, len(self.df))
        start = len(self.df) - true_length

        for i in range(start, len(self.df)):
            l = self.is_swing_low(i, GAP, col)
            if l:
                lows.append(l)

        return lows
        #======================================================

    #===========================
    #  Trend Checking
    #===========================

    def bg_trend_check(self,number_of_klines):
        #reset
        self.bg_trend_score = 0

        df = self.df.iloc[-number_of_klines:]

        macd_max_id = df["macd_signal"].idxmax()
        macd_max_high = df["macd_signal"].loc[macd_max_id]
        macd_min_id = df["macd_signal"].idxmin()
        macd_min_low = df["macd_signal"].loc[macd_min_id]
        macd_mean = df["macd"].mean()
        
        price_max_id = df["high"].idxmax()
        price_max_high = df["high"].loc[price_max_id]
        price_min_id = df["low"].idxmin()
        price_min_low = df["low"].loc[price_min_id]  
        price_mid = ((price_max_high + price_min_low)/2)
        price_mean = df["close"].mean()
        price_last = df["close"].iloc[-1]
        #print(price_mean)
        #print(price_mid)

        length = len(df)
        macd_positive_count = (df["macd_hist"] > 0).sum()
        macd_negative_count = (df["macd_hist"] < 0).sum()



        self.bg_trend_score += ((macd_positive_count/length)*100 - 50)
        #print(((macd_positive_count/length)*100 - 50))

        if macd_mean > 0:
            self.bg_trend_score += (macd_mean/macd_max_high) * 100 
            #print((macd_mean/macd_max_high) * 100)
        else:
            self.bg_trend_score -= (macd_mean/macd_min_low) * 100 
            #print(-(macd_mean/macd_min_low) * 100)

        if price_last > price_mid:
            self.bg_trend_score += ((price_last-price_mid)/(price_max_high-price_mid)) * 100 * 0.15
            #print(((price_last-price_mid)/(price_max_high-price_mid)) * 100 * 0.15) 
        else:
            self.bg_trend_score -= ((price_last-price_mid)/(price_min_low-price_mid)) * 100 * 0.15               
            #print(-((price_last-price_mid)/(price_min_low-price_mid)) * 100 * 0.15 )
        
        
        #print(self.bg_trend_score)

        return self.bg_trend_score

    #======================================================


    #==========================
    #   Gradient
    #=========================
    def Gradient(self):
        #reset
        self.low_gradient.clear()
        self.high_gradient.clear()
        self.low_trend_score.clear()
        self.high_trend_score.clear()
        self.trend_score = 0
        self.current_trend_score = 0
        diverge = 0
        total_score = 0
        
        price_max_id = self.df["high"].idxmax()
        price_max_high = self.df["high"].loc[price_max_id]
        price_min_id = self.df["low"].idxmin()
        price_min_low = self.df["low"].loc[price_min_id]
        self.price_range = price_max_high - price_min_low

        macd_max_id = self.df["macd_signal"].idxmax()
        macd_max_high = self.df["macd_signal"].loc[macd_max_id]
        macd_min_id = self.df["macd_signal"].idxmin()
        macd_min_low = self.df["macd_signal"].loc[macd_min_id]
        self.macd_range = macd_max_high - macd_min_low

        rsi_max_id = self.df[self.rsi].idxmax()
        rsi_max_high = self.df[self.rsi].loc[rsi_max_id]
        rsi_min_id = self.df[self.rsi].idxmin()
        rsi_min_low = self.df[self.rsi].loc[rsi_min_id]
        self.rsi_range = rsi_max_high - rsi_min_low

        #==================================================
        for i in range(1,len(self.big_highs)):
            price_gradient =  (self.big_highs[i]["price"] - self.big_highs[i-1]["price"])*60*5*100 / ((self.big_highs[i]["unix"] - self.big_highs[i-1]["unix"]) * self.price_range)
            macd_gradient =  (self.big_highs[i]["macd"] - self.big_highs[i-1]["macd"])*60*5*100 / ((self.big_highs[i]["unix"] - self.big_highs[i-1]["unix"]) * self.macd_range)
            rsi_gradient =  (self.big_highs[i]["rsi"] - self.big_highs[i-1]["rsi"])*60*5*100 / ((self.big_highs[i]["unix"] - self.big_highs[i-1]["unix"]) * self.rsi_range)
            
            if self.big_highs[i]["price"] > self.big_highs[i-1]["price"]:
                high_low = "higher"
            else:
                high_low = "lower"

            high_gradient = {
                "price" : price_gradient,
                "macd" : macd_gradient,
                "rsi" : rsi_gradient,
                "h1" : self.big_highs[i-1]["unix"],
                "h2" : self.big_highs[i]["unix"],
                "length" : self.big_highs[i]["unix"] - self.big_highs[i-1]["unix"],
                "high_low" : high_low
            }

            self.high_gradient.append(high_gradient)

        for i in range(1,len(self.big_lows)):
            price_gradient =  (self.big_lows[i]["price"] - self.big_lows[i-1]["price"])*60*5*100 / ((self.big_lows[i]["unix"] - self.big_lows[i-1]["unix"]) * self.price_range)
            macd_gradient =  (self.big_lows[i]["macd"] - self.big_lows[i-1]["macd"])*60*5*100 / ((self.big_lows[i]["unix"] - self.big_lows[i-1]["unix"]) *self. macd_range)
            rsi_gradient =  (self.big_lows[i]["rsi"] - self.big_lows[i-1]["rsi"])*60*5*100 / ((self.big_lows[i]["unix"] - self.big_lows[i-1]["unix"]) * self.rsi_range)

            if self.big_lows[i]["price"] > self.big_lows[i-1]["price"]:
                high_low = "higher"
            else:
                high_low = "lower"

            low_gradient = {
                "price" : price_gradient,
                "macd" : macd_gradient,
                "rsi" : rsi_gradient,
                "l1" : self.big_lows[i-1]["unix"],
                "l2" : self.big_lows[i]["unix"],
                "length" : self.big_lows[i]["unix"] - self.big_lows[i-1]["unix"],
                "high_low" : high_low
            }

            self.low_gradient.append(low_gradient)

        #print("low: ",self.low_gradient)
        print("")
        #print("high: ",self.high_gradient)

        #=================================================
        for i in range(0,len(self.low_gradient)):
            low = self.low_gradient[i]

            diverge = (low["macd"] + low["rsi"] - 2 * low["price"]) /2      
            trend = (low["price"] * 2 + low["macd"] + low["rsi"]) /4
            total_score = diverge + trend

            low_trend_score = {
                "背驰分" :diverge ,
                "趋势分" :trend ,
                "total" : total_score,
                "l1" : low["l1"],
                "l2" : low["l2"],
                "length" : low["length"],
                "high_low" : low["high_low"]
            }

            self.low_trend_score.append(low_trend_score)

        for i in range(0,len(self.high_gradient)):
            high = self.high_gradient[i]

            diverge = (high["macd"] + high["rsi"] - 2 * high["price"]) / 2    
            trend = (high["price"] * 2 + high["macd"] + high["rsi"]) /4
            total_score = diverge + trend

            high_trend_score = {
                "背驰分" :diverge ,
                "趋势分" :trend ,
                "total" : total_score,
                "h1" : high["h1"],
                "h2" : high["h2"],
                "length" : high["length"],
                "high_low" : high["high_low"]
            }

            self.high_trend_score.append(high_trend_score)

        print("")
        for id, item in enumerate(self.low_trend_score):
          print(f"low策略分{id+1}: ",round(item["背驰分"],3),"+",round(item["趋势分"],3) ,"=", round(item["total"],3),item["high_low"])
        #print("")
        for id, item in enumerate(self.high_trend_score):
          print(f"high策略分{id+1}: ",round(item["背驰分"],3),"+",round(item["趋势分"],3) ,"=", round(item["total"],3),item["high_low"])


    #=======================================================

    #===========================
    #    Current Trend Score
    
        current_unix = int(self.df.index[-1].timestamp())
        # clear if y <= 0
        self.current_trend_list = [ item for item in self.current_trend_list if item["y"] > 0 ]

        if self.Trend_decision > 0:
            for high in self.high_trend_score:
                exist = False
                if len(self.current_trend_list) > 0:
                    for item in self.current_trend_list:
                        if high["h2"] <= item["unix"]:
                            exist = True
                            break
                    
                    if exist:
                        continue

                x = current_unix - high["h2"]
                y = - (x / (high["length"] * 1.5))**5 + 1
                current_score = y * high["total"]

                current_list = {
                    "unix" : high["h2"],
                    "y" : y,
                    "length" : high["length"],
                    "maximum" : high["total"],
                    "current_score" : current_score
                }

                self.current_trend_list.append(current_list)

        if self.Trend_decision < 0:
            for low in self.low_trend_score:
                exist = False
                if len(self.current_trend_list) > 0:
                    for item in self.current_trend_list:
                        if low["l2"] <= item["unix"]:
                            exist = True
                            break
                    
                    if exist:
                        continue

                x = current_unix - low["l2"]
                y = - (x / (low["length"] * 1.5))**5 + 1
                current_score = y * low["total"]

                current_list = {
                    "unix" : low["l2"],
                    "y" : y,
                    "length" : low["length"],
                    "maximum" : low["total"],
                    "current_score" : current_score
                }

                self.current_trend_list.append(current_list)
        
        for item in self.current_trend_list:
            x = current_unix - item["unix"]
            y = - (x / (item["length"] * 1.5))**5 + 1
            item["y"] = y
            item["current_score"] = y * item["maximum"]           


        self.current_trend_list = [ item for item in self.current_trend_list if item["y"] > 0 ]

        for item in self.current_trend_list:
            self.current_trend_score += item["current_score"]

        #print("List: ",self.current_trend_list)
        print("Score: ",self.current_trend_score)
                
        # ===== 先根据分数判定当前趋势 =====
        if self.Trend_decision > 10:
            self.trend_name = "LONG"
        elif self.Trend_decision < -10:
            self.trend_name = "SHORT"
        #else:
            #new_trend = None   # 中性 / 无趋势


        # ===== 判断是否发生趋势反转 =====
        if len(self.low_trend_score) > 0:
            if self.low_trend_score[-1]["high_low"] == "lower":
                self.trend_inverse = False

        if len(self.high_trend_score) > 0:
            if self.high_trend_score[-1]["high_low"] == "higher":
                self.trend_inverse = False

        if self.prev_trend_name is not None and self.trend_name is not None:
            if self.prev_trend_name != self.trend_name:
                self.trend_inverse = {
                    "bool": True,
                    "time": self.df.index[-1],                  # ✅ pd.Timestamp
                    "unix": int(self.df.index[-1].timestamp()), # ✅ int
                    "from": self.prev_trend_name,
                    "to": self.trend_name
                }


        # ===== 更新当前趋势 =====
        self.prev_trend_name = self.trend_name
        print(self.trend_inverse)
    #======================================================




    #=====================================================

    #==========================
    #     Strategy
    #==========================
    def Strategy(self):
        #reset
        self.swing_highs.clear()
        self.swing_lows.clear()
        self.big_highs.clear()
        self.big_lows.clear()




        for i in range(len(self.df)):
            high = self.is_swing_high(i, 10, "close")
            if high:
                self.swing_highs.append(high)
            low = self.is_swing_low(i, 10, "close")
            if low:
                self.swing_lows.append(low)

        #self.big_highs = self.unique_swing_highs(start_gap=20, max_gap=150, col="close",num_high=3,length=150)
        #self.big_lows = self.unique_swing_lows(start_gap=20, max_gap=150, col="close",num_low=3,length=150)

        self.big_highs = self.fixed_swing_highs_20(col="close")
        self.big_lows = self.fixed_swing_lows_20(col="close")


        self.latest_high = self.find_latest_swing_high(n=3, col="close")
        self.latest_low  = self.find_latest_swing_low(n=3, col="close")

        #print(self.big_highs[0]["unix"])
        #print(self.big_highs[1]["unix"])
        #print(self.big_highs[2]["unix"])      


        bg_score_80 = self.bg_trend_check(50)

        bg_score_180 = self.bg_trend_check(180)
        print("80: ",bg_score_80)
        print("180: ",bg_score_180)
        self.Trend_decision = bg_score_80 * 0.5 + bg_score_180
        print(self.Trend_decision)

        self.Gradient()  
        
        #======================================
        #          Backtest
        #======================================

        #  Entry
        if self.position_state == 0:
            pass

        # Exit
        if self.position_state == 1:
            pass
    #=========================================================



    #========================
    #    Plot Graph
    #========================
    def plot_graph(self):

        # ========= 1️⃣ 创建空 Series =========
        buy_point = pd.Series(np.nan, index=self.df.index)
        sell_point = pd.Series(np.nan, index=self.df.index)

        swing_high_series = pd.Series(np.nan, index=self.df.index)
        swing_low_series = pd.Series(np.nan, index=self.df.index)

        swing_high_latest = pd.Series(np.nan, index=self.df.index)
        swing_low_latest = pd.Series(np.nan, index=self.df.index)

        big_high_series = pd.Series(np.nan, index=self.df.index)
        big_low_series = pd.Series(np.nan, index=self.df.index)

        rsi_high_series = pd.Series(np.nan, index=self.df.index)
        rsi_low_series  = pd.Series(np.nan, index=self.df.index)

        macd_high_series = pd.Series(np.nan, index=self.df.index)
        macd_low_series  = pd.Series(np.nan, index=self.df.index)  

        macd_hist_pos = self.df['macd_hist'].where(self.df['macd_hist'] >= 0)
        macd_hist_neg = self.df['macd_hist'].where(self.df['macd_hist'] < 0)

        if len(self.entry) > 0:
            buy_point.loc[self.entry[-1]["time"]] = self.entry[-1]["price"]
        if len(self.exit) > 0:
            sell_point.loc[self.exit[-1]["time"]] = self.exit[-1]["price"]

        # ========= 2️⃣ 把 swing 填进去 =========
        for s in self.swing_highs:
            swing_high_series.loc[s["time"]] = s["price"]
        for s in self.swing_lows:
            swing_low_series.loc[s["time"]] = s["price"]

        for s in self.big_highs:
            big_high_series.loc[s["time"]] = s["price"]
            rsi_high_series.loc[s["time"]] = s["rsi"]
            macd_high_series.loc[s["time"]] = s["macd"]

        for s in self.big_lows:
            big_low_series.loc[s["time"]] = s["price"]
            rsi_low_series.loc[s["time"]] = s["rsi"]
            macd_low_series.loc[s["time"]] = s["macd"]

        swing_high_latest.loc[self.latest_high["time"]] = self.latest_high["price"]
        swing_low_latest.loc[self.latest_low["time"]] = self.latest_low["price"]
        # ========= addplot =========
        add = []

        #========= Entry Exit Point ================
        if len(self.entry) > 0:
            add.append(
                mpf.make_addplot(
                    buy_point,
                    type='scatter',
                    marker='^',
                    markersize=120,
                    color='green'
                )
            )
        if len(self.exit) > 0:
            add.append(
                mpf.make_addplot(
                    sell_point,
                    type='scatter',
                    marker='v',
                    markersize=120,
                    color='red'
                )
            )
        #============ big swing ==================
        if len(self.big_highs) > 0:
            add.append(mpf.make_addplot(big_high_series,type='scatter',marker='v',markersize=100,color='purple'))
        else: 
            print("Not enough high")
        if len(self.big_lows) > 0: 
            add.append(mpf.make_addplot(big_low_series,type='scatter',marker='^',markersize=100,color='orange'))
        else:
            print("Not enough low")
      
        #================ 🔴 swing high 红点 ============= 
        #add.append(mpf.make_addplot(swing_high_series,type='scatter',marker='v',markersize=60,color='blue'))
        #add.append(mpf.make_addplot(swing_low_series,type='scatter',marker='^',markersize=60,color='red'))
        add.append(mpf.make_addplot(swing_high_latest,type='scatter',marker='v',markersize=40,color='green'))
        add.append(mpf.make_addplot(swing_low_latest,type='scatter',marker='^',markersize=40,color='red'))


        # =============== MACD panel (panel=1) ==========
        add.append(mpf.make_addplot(self.df['macd'],panel=1,color='blue',ylabel='MACD'))
        add.append(mpf.make_addplot(self.df['macd_signal'],panel=1,color='orange'))
        
        add.append(mpf.make_addplot(macd_hist_pos,type='bar' ,panel=1,color='green'))
        add.append(mpf.make_addplot(macd_hist_neg,type='bar' ,panel=1,color='red'))

        if macd_low_series.notna().any():
            add.append(mpf.make_addplot(macd_low_series,panel=1,type='scatter',marker='^',markersize=40,color='green'))
        if macd_high_series.notna().any():
            add.append(mpf.make_addplot(macd_high_series,panel=1,type='scatter',marker='v',markersize=40,color='red'))

        #=============== RSI panel (panel=2==============
        #add.append(mpf.make_addplot(self.df['rsi8'],panel=2,color='purple',ylabel='RSI'))
        add.append(mpf.make_addplot(self.df['rsi14'],panel=2,color='green'))
        # 🔺 RSI swing
        if rsi_high_series.notna().any():
            add.append(mpf.make_addplot(rsi_high_series,panel=2,type='scatter',marker='v',markersize=40,color='red'))
        if rsi_low_series.notna().any():
            add.append(mpf.make_addplot(rsi_low_series,panel=2,type='scatter',marker='^',markersize=40,color='green'))

        #======= EMA ====================
        #add.append(mpf.make_addplot(self.df['ema8'], color='orange', width=1.5, linestyle='solid', ylabel='EMA50'))
        #add.append(mpf.make_addplot(self.df['ema26'], color='purple', width=1.5, linestyle='solid', ylabel='EMA100'))
   
        # ========= 主绘图 =========
        fig, axes = mpf.plot(
            self.df,
            type='candle',
            style='charles',
            volume=False,
            addplot=add,
            panel_ratios=(4, 1, 1),
            figratio=(14, 8),
            datetime_format='%Y-%m-%d',
            returnfig=True
        )

        # ========= panel 分隔线 =========
        fig.subplots_adjust(hspace=1)  # 数字越大，中间间距越大

        for ax in axes[1:]:
            ax.spines['top'].set_visible(True)
            ax.spines['top'].set_color('black')
            ax.spines['top'].set_linewidth(1.2)

        plt.show()

        #==============================



#========================
#      Initiallize 
#========================
bitcoin = Bitcoin()

start_time="2025-01-20 00:00:00"
end_time="2025-12-21 10:00:00"
symbol = "BTCUSDT"
#=========================

#=========================
#       Main
#=========================
bitcoin = Bitcoin()
bitcoin.random_klines("BTCUSDT",
                       Client.KLINE_INTERVAL_15MINUTE,
                        start_time,
                        end_time)
bitcoin.fix_duplicate_price(col="close", max_gap=10, offset=0.1)

bitcoin.add_ema(8)
bitcoin.add_ema(26)
bitcoin.add_ema(200)
bitcoin.add_rsi(14)
bitcoin.add_macd()
bitcoin.Strategy()
bitcoin.plot_graph()

# 开始监听 Space 增量
while True:
    if keyboard.is_pressed("alt"):
        bitcoin.live_update("BTCUSDT", Client.KLINE_INTERVAL_15MINUTE,5) 
    if keyboard.is_pressed("ctrl"):
        bitcoin.live_update("BTCUSDT", Client.KLINE_INTERVAL_15MINUTE,20) 
    if keyboard.is_pressed("tab"):
        bitcoin.live_update("BTCUSDT", Client.KLINE_INTERVAL_15MINUTE,50) 
    if keyboard.is_pressed("esc"):
        

        break