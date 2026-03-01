import pandas as pd
import mplfinance as mpf
import datetime
from binance.client import Client
import time
import ta  
import numpy as np
import matplotlib.pyplot as plt

# ========================
# Binance Client
# ========================
try:
    client = Client()
except Exception as e:
    print("Binance API unreachable:", e)


class Bitcoin():
    def __init__(self):
        self.df = pd.DataFrame()
        #====== parameter ======
        self.macd_greater_threshold = 1.15
        self.macd_window = 28  #  macd peak 的 window
        self.sma_window = 90   # 判断 macd long 还是 short 的 window
        self.ema_window = 140  # 0.5 以上的 止损线
        self.entry_rsi = 37  # 入场rsi
        self.exit_rsi = 75
        self.tp_ratio = 0.15     # 0 是前高， 0.1是 比前高还多10%
        self.sl_ratio = 0.85    # 1 是上涨起点 0.8 是小过起点
        self.invalid_drawback = 0.85
        self.required_bull_count = 3
        self.required_bear_count = 3
        

        #====== initialize =====
        self.cummulative_profit = 0

        self.line = []
        self.mse = 0
        self.slope = 0
        self.up_percent = 0
        self.drawdown = 0
        self.tp_count = 0
        self.sl_count = 0

        self.entry = False
        self.check_macd = None
        self.buy_criteria = None
        self.head_tail_range = 0
        self.volume_count = 0
        self.bg_trend_score = 0   # 用来存最近 200 根背景趋势分数

        self.pre_macd_top = None
        self.pre_macd_bottom = None
        self.macd_top = None
        self.macd_bottom = None

        self.macd_greater= False
        self.trend_side = "SKIP"
        self.Long_pre1 = False
        self.Long_pre2 = False
        self.Short_pre1 = False
        self.Short_pre2 = False
        self.entry_price = 0
        self.entry_index = 0
        self.position = None  # LONG / SHORT
        self.long_invalid_price = None
        self.short_invalid_price = None
        self.long_tp = None
        self.long_sl = None
        self.short_tp = None
        self.short_sl = None
    # ==================
    # Indicator
    # ==================
    def add_ema(self):
        self.df[f'ema{self.ema_window}'] = ta.trend.EMAIndicator(
            self.df["close"], window=self.ema_window
        ).ema_indicator()

    def add_sma(self):
        self.df[f'sma{self.sma_window}'] = self.df["close"].rolling(self.sma_window).mean()

    def add_rsi(self, window):
        self.df[f'rsi{window}'] = ta.momentum.RSIIndicator(
            self.df["close"], window=window
        ).rsi()

    def add_macd(self, fast=12, slow=26, signal=9):
        macd = ta.trend.MACD(
            self.df['close'],
            window_fast=fast,
            window_slow=slow,
            window_sign=signal
        )
        self.df['macd'] = macd.macd()
        self.df['macd_signal'] = macd.macd_signal()
        self.df['macd_hist'] = macd.macd_diff()
    

    def add_macd_peak(self, price_window=5):

        if 'macd' not in self.df.columns:
            raise ValueError("Run add_macd() first.")

        macd = self.df['macd']
        window = self.macd_window * 2 + 1

        rolling_max = macd.rolling(window=window, center=True).max()
        rolling_min = macd.rolling(window=window, center=True).min()

        is_top = macd == rolling_max
        is_bottom = macd == rolling_min

        self.df['macd_top'] = np.where(is_top, macd, None)
        self.df['macd_bottom'] = np.where(is_bottom, macd, None)

        # 价格列（你可以改成 close）
        high = self.df['close']
        low = self.df['close']

        # 初始化
        self.df['macd_top_price'] = None
        self.df['macd_bottom_price'] = None

        # ===== 处理 TOP =====
        top_indices = self.df.index[is_top]

        for idx in top_indices:

            loc = self.df.index.get_loc(idx)

            start = max(0, loc - price_window)
            end = min(len(self.df) - 1, loc + price_window)

            price_slice = high.iloc[start:end+1]
            real_high = price_slice.max()

            # 🔥 存回 MACD peak 那根
            self.df.loc[idx, 'macd_top_price'] = real_high


        # ===== 处理 BOTTOM =====
        bottom_indices = self.df.index[is_bottom]

        for idx in bottom_indices:

            loc = self.df.index.get_loc(idx)

            start = max(0, loc - price_window)
            end = min(len(self.df) - 1, loc + price_window)

            price_slice = low.iloc[start:end+1]
            real_low = price_slice.min()

            # 🔥 存回 MACD peak 那根
            self.df.loc[idx, 'macd_bottom_price'] = real_low

    def check_latest_macd_peak(self, price_window=5):
        """
        每次调用只检查倒数第1 - macd_window的K线是否是MACD peak。
        如果是peak，返回 dict:
        {"peak": "top"/"bottom", "index": index, "macd": macd值, "price":价格}
        """
        if len(self.df) <= self.macd_window * 2:
            return None  # 数据太少

        pivot_idx = -1 - self.macd_window  # 倒数第1 - macd_window
        pivot_macd = self.df["macd"].iloc[pivot_idx]

        # +/- macd_window 判断局部 max/min
        start = max(0, pivot_idx - self.macd_window)
        end = min(len(self.df), pivot_idx + self.macd_window + 1)
        macd_slice = self.df["macd"].iloc[start:end]

        # +/- price_window 找对应价格
        price_start = max(0, pivot_idx - price_window)
        price_end = min(len(self.df), pivot_idx + price_window + 1)
        price_slice = self.df["close"].iloc[price_start:price_end]

        # 判断 MACD TOP
        if pivot_macd == macd_slice.max():
            return {
                "peak": "top",
                "index": self.df.index[pivot_idx],
                "macd": pivot_macd,
                "price": price_slice.max()
            }

        # 判断 MACD BOTTOM
        if pivot_macd == macd_slice.min():
            return {
                "peak": "bottom",
                "index": self.df.index[pivot_idx],
                "macd": pivot_macd,
                "price": price_slice.min()
            }

        return None  # 没有peak

    def add_volume_ma(self, window):
        self.df[f'vol_ma'] = self.df['volume'].rolling(window).mean()

    # ==================
    # Get Klines
    # ==================
    def get_history_klines(self, symbol, interval, start_str, end_str=None):

        df_all = []

        start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_str).timestamp() * 1000) if end_str else None

        while True:
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

            start_ts = klines[-1][6] + 1

            if end_ts and start_ts >= end_ts:
                break

        df = pd.concat(df_all, ignore_index=True)

        for col in ["open","high","low","close","volume"]:
            df[col] = df[col].astype(float)

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)

        return df
    
    # ======================
    # Klines
    # ======================
    def get_latest_klines(self):
        klines = client.get_klines(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_15MINUTE, limit=250)
        df = pd.DataFrame(klines, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","qav","trades","tb","tq","ignore"
        ])
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)

        df["time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("time", inplace=True)
        self.df = df

    
# ==================
# Strategy
# ==================
    def Strategy(self):


        timestamp = self.df.index[i] + pd.Timedelta(hours=8)


        # ===== Top =======
        if pd.notna(self.df["macd_top"].iloc[i]):

            # pivot 所在 index
            pivot_loc = self.df.index.get_loc(self.df.index[i])

            start = max(0, pivot_loc - (self.macd_window+0))
            end = pivot_loc

            # 往前 window 根K 找最低价
            price_slice = self.df["low"].iloc[start:end+1]
            start_price = price_slice.min()

            macd_top = {
                "i": i,
                "index": self.df.index[i],
                "macd": self.df["macd_top"].iloc[i],
                "price": self.df["macd_top_price"].iloc[i],
                "sma": self.df[f"sma{self.sma_window}"].iloc[i],
                "start_price": start_price
            }

            # === 计算 80% 回撤失效价 ===
            swing_size = macd_top["price"] - macd_top["start_price"]

            if swing_size > 0:
                self.long_invalid_price = macd_top["price"] - swing_size * self.invalid_drawback
            else:
                self.long_invalid_price = None

            # === greater than the previous macd peak , and reject extreme macd
            if macd_top["macd"] > self.pre_macd_top["macd"] * self.macd_greater_threshold and macd_top["macd"] < macd_top["price"] * 0.015:
                self.macd_greater= True
            else:
                self.macd_greater= False

            self.pre_macd_top = macd_top

            if abs(macd_top["price"] - macd_top["sma"]) > abs(self.pre_macd_bottom["price"] - self.pre_macd_bottom["sma"]):
                self.trend_side = "LONG"
            else: 
                self.trend_side = "SKIP"

            #print("Top   : ",self.df.index[i], self.df["macd_top"].iloc[i], self.df["macd_top_price"].iloc[i],macd_greater,self.trend_side)

        # ======== Bottom ========
        if pd.notna(self.df["macd_bottom"].iloc[i]):

            pivot_loc = self.df.index.get_loc(self.df.index[i])

            start = max(0, pivot_loc - (self.macd_window+0))
            end = pivot_loc

            # 往前 window 根K 找最高价
            price_slice = self.df["high"].iloc[start:end+1]
            start_price = price_slice.max()

            macd_bottom = {
                "i": i,
                "index": self.df.index[i],
                "macd": self.df["macd_bottom"].iloc[i],
                "price": self.df["macd_bottom_price"].iloc[i],
                "sma": self.df[f"sma{self.sma_window}"].iloc[i],
                "start_price": start_price
            }

            swing_size = macd_bottom["start_price"] - macd_bottom["price"]

            if swing_size > 0:
                self.short_invalid_price = macd_bottom["price"] + swing_size * self.invalid_drawback
            else:
                self.short_invalid_price = None

            
            # === greater than the previous macd peak, reject extreme macd
            if macd_bottom["macd"] < self.pre_macd_bottom["macd"] * self.macd_greater_threshold and macd_bottom["macd"] > -macd_bottom["price"] * 0.015:
                self.macd_greater= True
            else:
                self.macd_greater= False

            self.pre_macd_bottom = macd_bottom

            if abs(macd_bottom["price"] - macd_bottom["sma"]) > abs(self.pre_macd_top["price"] - self.pre_macd_top["sma"]):
                self.trend_side = "SHORT"
            else: 
                self.trend_side = "SKIP"

            #print("Bottom: ",self.df.index[i], self.df["macd_bottom"].iloc[i], self.df["macd_bottom_price"].iloc[i],macd_greater,self.trend_side)

        macd_peak = self.check_latest_macd_peak()

        if macd_peak is not None:
            if macd_peak["peak"] == "top":
                if self.macd_top is not None:
                    self.pre_macd_top = self.macd_top
                
                

                start = -1 - self.macd_window * 2
                end = -1 - self.macd_window

                price_slice = self.df["low"].iloc[start:end+1]
                start_price = price_slice.min()

                self.macd_top = {
                    "index": self.df.index[-1-self.macd_window],
                    "macd": macd_peak["macd"],
                    "price": macd_peak["price"],
                    "start_price": start_price,
                    "sma": self.df[f"sma{self.sma_window}"].iloc[-1-self.macd_window],
                }

            if macd_peak["peak"] == "bottom":
                if self.macd_bottom is not None:
                    self.pre_macd_bottom = self.macd_bottom
                
                self.macd_bottom = macd_peak



        # ======================
        # 计算当前连阳数量
        # ======================
        long_streak = 0
        for i in range(1, len(self.df)+1):  # 从最新往前遍历
            if self.df.iloc[-i]["close"] > self.df.iloc[-i]["open"]:
                long_streak += 1
            else:
                break

        # ======================
        # 计算当前连阴数量
        # ======================
        short_streak = 0
        for i in range(1, len(self.df)+1):
            if self.df.iloc[-i]["close"] < self.df.iloc[-i]["open"]:
                short_streak += 1
            else:
                break


        # ===================
        # 入场逻辑
        # ===================
        if not self.entry:
            # ===== LONG 80% invalidation =====
            if self.Long_pre1 and self.long_invalid_price is not None:
                if self.df["close"].iloc[i] <= self.long_invalid_price:
                    self.Long_pre1 = False
                    self.Long_pre2 = False
                    self.macd_greater= False
                    self.trend_side = "SKIP"      # ⭐ 加这一行
            # ===== SHORT 80% invalidation =====
            if self.Short_pre1 and self.short_invalid_price is not None:
                if self.df["close"].iloc[i] >= self.short_invalid_price:
                    self.Short_pre1 = False
                    self.Short_pre2 = False
                    self.macd_greater= False
                    self.trend_side = "SKIP"      # ⭐ 加这一行

            # -------- 第一层 --------
            if self.macd_greater and self.trend_side == "LONG":
                self.Long_pre1 = True
                self.Short_pre1 = False

            elif self.macd_greater and self.trend_side == "SHORT":
                self.Short_pre1 = True
                self.Long_pre1 = False


            # -------- 第二层 RSI --------
            if self.Long_pre1:
                if self.df["rsi14"].iloc[i] < self.entry_rsi:
                    self.Long_pre2 = True

            if self.Short_pre1:
                if self.df["rsi14"].iloc[i] > 100 - self.entry_rsi:
                    self.Short_pre2 = True


            # -------- 第三层 K线触发 --------
            # 连续2根 bullish
            if self.Long_pre2:
                if long_streak >= self.required_bull_count:
                    
                    self.entry = "LONG"
                    self.position = "LONG"
                    self.entry_price = self.df["close"].iloc[i]
                    self.entry_index = i

                    # ===== 计算止盈止损 =====
                    swing_high = self.pre_macd_top["price"]
                    start_price = self.pre_macd_top["start_price"]
                    retracement = (swing_high - self.entry_price) / (swing_high - start_price)

                    if retracement <= 0.5 and self.entry_price > self.df[f"ema{self.ema_window}"].iloc[i]: 
                        self.long_sl = self.df[f"ema{self.ema_window}"].iloc[i]
                        self.long_tp = swing_high + (swing_high - start_price) * self.tp_ratio
                    else:
                        self.long_sl = swing_high - (swing_high - start_price) * self.sl_ratio
                        self.long_tp = swing_high
                    

                    print(self.df.index[i], "🚀 LONG ENTRY", self.df["close"].iloc[i], "| TP: ",self.long_tp,"| SL: ",self.long_sl,"| Retracement: ",round(retracement,2))

                    # reset
                    self.Long_pre1 = False
                    self.Long_pre2 = False
                    self.macd_greater= False


            # 连续2根 bearish
            if self.Short_pre2:
                if short_streak >= self.required_bear_count:
                    
                    self.entry = "SHORT"
                    self.position = "SHORT"
                    self.entry_price = self.df["close"].iloc[i]
                    self.entry_index = i

                    # 计算 SHORT 止盈止损
                    swing_low = self.pre_macd_bottom["price"]
                    start_price = self.pre_macd_bottom["start_price"]
                    retracement = (self.entry_price - swing_low) / (start_price - swing_low)
                    
                    if retracement <= 0.5 and self.entry_price < self.df[f"ema{self.ema_window}"].iloc[i]:
                        self.short_sl = self.df[f"ema{self.ema_window}"].iloc[i]
                        self.short_tp = swing_low + (swing_low - start_price) * self.tp_ratio
                    else:
                        self.short_sl = swing_low - (swing_low - start_price) * self.sl_ratio
                        self.short_tp = swing_low

                    

                    print(self.df.index[i], "🔻 SHORT ENTRY", self.df["close"].iloc[i], "| TP: ",self.short_tp,"| SL: ",self.short_sl,"| Retracement: ",round(retracement,2))
                    # reset
                    self.Short_pre1 = False
                    self.Short_pre2 = False
                    self.macd_greater = False
            

        # ===================
        # 持仓 / 出场逻辑
        # ===================
        if self.entry == "LONG":
            close = self.df["close"].iloc[i]
            tp_hit = close >= self.long_tp
            sl_hit = close <= self.long_sl
            rsi_exit = self.df["rsi14"].iloc[i] > self.exit_rsi

            if tp_hit or sl_hit or rsi_exit:
                exit_price = close
                pnl_pct = (exit_price - self.entry_price) / self.entry_price * 100
                self.cummulative_profit += pnl_pct - 0.04

                if pnl_pct > 0:
                    self.tp_count += 1
                else:
                    self.sl_count += 1

                # 组合 exit reason
                reasons = []
                if tp_hit:
                    reasons.append("TP")
                if sl_hit:
                    reasons.append("SL")
                if rsi_exit:
                    reasons.append("RSI>70")
                exit_reason = "/".join(reasons)

                print(self.df.index[i], "❌ LONG EXIT", exit_price,
                    "| PnL:", round(pnl_pct,2), "%", "| Reason:", exit_reason)
                print("CUM:", round(self.cummulative_profit,2), "%\n")

                self.entry = False
                self.position = None

        if self.entry == "SHORT":
            close = self.df["close"].iloc[i]
            tp_hit = close <= self.short_tp
            sl_hit = close >= self.short_sl
            rsi_exit = self.df["rsi14"].iloc[i] < 100 - self.exit_rsi

            if tp_hit or sl_hit or rsi_exit:
                exit_price = close
                pnl_pct = (self.entry_price - exit_price) / self.entry_price * 100
                self.cummulative_profit += pnl_pct - 0.04

                if pnl_pct > 0:
                    self.tp_count += 1
                else:
                    self.sl_count += 1

                # 组合 exit reason
                reasons = []
                if tp_hit:
                    reasons.append("TP")
                if sl_hit:
                    reasons.append("SL")
                if rsi_exit:
                    reasons.append("RSI<30")
                exit_reason = "/".join(reasons)

                print(self.df.index[i], "❌ SHORT EXIT", exit_price,
                    "| PnL:", round(pnl_pct,2), "%", "| Reason:", exit_reason)
                print("CUM:", round(self.cummulative_profit,2), "%\n")

                self.entry = False
                self.position = None


bitcoin = Bitcoin()

start_time = "2025-01-01 00:00:00"
end_time   = "2026-02-26 10:00:00"
symbol     = "BTCUSDT"
'''
bitcoin.df = bitcoin.get_history_klines(
    symbol=symbol,
    interval=Client.KLINE_INTERVAL_15MINUTE,
    start_str=start_time,
    end_str=end_time
)
#'''

bitcoin.get_latest_klines()
#bitcoin.add_ema(8)
#bitcoin.add_ema(26)
bitcoin.add_ema()
bitcoin.add_sma()
bitcoin.add_rsi(14)
bitcoin.add_macd()
bitcoin.add_macd_peak()
bitcoin.add_volume_ma(10)


bitcoin.Strategy()


