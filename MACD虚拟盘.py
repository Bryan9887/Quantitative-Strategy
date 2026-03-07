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

        self.macd_greater_long = False
        self.macd_greater_short= False
        self.trend_side = "SKIP"
        self.Long_pre1 = False
        self.Long_pre2 = False
        self.Short_pre1 = False
        self.Short_pre2 = False
        self.entry_price = 0
        self.entry_index = 0
        self.position = None  # LONG / SHORT
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
        #self.df["vol_ma"] = self.df["volume"].rolling(10).mean()

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
        start = pivot_idx - self.macd_window
        end = pivot_idx + self.macd_window
        
        macd_slice = self.df["macd"].iloc[start:end]

        # +/- price_window 找对应价格
        price_start = pivot_idx - price_window
        price_end = pivot_idx + price_window
        price_slice = self.df["close"].iloc[price_start:price_end]

        #print("Max:",round(macd_slice.max(),1),"Min:",round(macd_slice.min(),1),"Pivot:",round(pivot_macd,1))
        print("Pivot Time: ",self.df.index[pivot_idx])
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

        macd_peak = self.check_latest_macd_peak()

        pivot_idx = -1 - self.macd_window  # 倒数第1 - macd_window

        if macd_peak is not None:
            if macd_peak["peak"] == "top":

                # 把旧的save去previous
                if self.macd_top is not None and self.df.index[pivot_idx] != self.macd_top["index"]:
                    self.pre_macd_top = self.macd_top
                    print(self.pre_macd_top)
                

                start = -1 - self.macd_window * 2
                end = -1 - self.macd_window

                price_slice = self.df["close"].iloc[start:end+1]
                start_price = price_slice.min()

                self.macd_top = {
                    "index": self.df.index[-1-self.macd_window],
                    "macd": macd_peak["macd"],
                    "price": macd_peak["price"],
                    "start_price": start_price,
                    "sma": self.df[f"sma{self.sma_window}"].iloc[-1-self.macd_window],
                    "invalid_price": macd_peak["price"]  - self.invalid_drawback * (macd_peak["price"] - start_price)
                }

                '''
                print(
                    f'| Macd_Top:{round(self.macd_top["macd"],1)} '
                    f'| Price:{round(self.macd_top["price"],1)} '
                    f'| Start_Price:{round(self.macd_top["start_price"],1)} '
                    f'| SMA:{round(self.macd_top["sma"],1)} '
                    f'| Invalid:{round(self.macd_top["invalid_price"],1)} '
                )
                #'''

            if macd_peak["peak"] == "bottom":
                if self.macd_bottom is not None and self.df.index[pivot_idx] != self.macd_bottom["index"]:
                    self.pre_macd_bottom = self.macd_bottom
                    print(self.pre_macd_bottom)
                
                start = -1 - self.macd_window * 2
                end = -1 - self.macd_window

                price_slice = self.df["close"].iloc[start:end+1]
                start_price = price_slice.max()

                self.macd_bottom = {
                    "index": self.df.index[-1-self.macd_window],
                    "macd": macd_peak["macd"],
                    "price": macd_peak["price"],
                    "start_price": start_price,
                    "sma": self.df[f"sma{self.sma_window}"].iloc[-1-self.macd_window],
                    "invalid_price": macd_peak["price"]  + self.invalid_drawback * (macd_peak["price"] - start_price)
                }

                '''
                print(
                    f'| Macd_Bottom:{round(self.macd_bottom["macd"],1)} '
                    f'| Price:{round(self.macd_bottom["price"],1)} '
                    f'| Start_Price:{round(self.macd_bottom["start_price"],1)} '
                    f'| SMA:{round(self.macd_bottom["sma"],1)} '
                    f'| Invalid:{round(self.macd_bottom["invalid_price"],1)} '
                )
                #'''


            
        # 开始判断，前提是 4 个都不是none才可以开始策略
        if self.pre_macd_bottom is not None and self.pre_macd_top is not None and self.macd_bottom is not None and self.macd_top is not None:
            print(
                f'| Macd_Top:{round(self.macd_top["macd"],1)} '
                f'| Price:{round(self.macd_top["price"],1)} '
                f'| Start_Price:{round(self.macd_top["start_price"],1)} '
                f'| SMA:{round(self.macd_top["sma"],1)} '
                f'| Invalid:{round(self.macd_top["invalid_price"],1)} '
            )

            print(
                f'| Macd_Bottom:{round(self.macd_bottom["macd"],1)} '
                f'| Price:{round(self.macd_bottom["price"],1)} '
                f'| Start_Price:{round(self.macd_bottom["start_price"],1)} '
                f'| SMA:{round(self.macd_bottom["sma"],1)} '
                f'| Invalid:{round(self.macd_bottom["invalid_price"],1)} '
            )

            print(
                f'| Pre_Macd_Top:{round(self.pre_macd_top["macd"],1)} '
                f'| Price:{round(self.pre_macd_top["price"],1)} '
                f'| Start_Price:{round(self.pre_macd_top["start_price"],1)} '
                f'| SMA:{round(self.pre_macd_top["sma"],1)} '
                f'| Invalid:{round(self.pre_macd_top["invalid_price"],1)} '
            )
            print(
                f'| Pre_Macd_Bottom:{round(self.pre_macd_bottom["macd"],1)} '
                f'| Price:{round(self.pre_macd_bottom["price"],1)} '
                f'| Start_Price:{round(self.pre_macd_bottom["start_price"],1)} '
                f'| SMA:{round(self.pre_macd_bottom["sma"],1)} '
                f'| Invalid:{round(self.pre_macd_bottom["invalid_price"],1)} '
            )
        
            # 最新的为多的macd
            if self.macd_top["index"] > self.macd_bottom["index"]:

                # === greater than the previous macd peak , and reject extreme macd
                if self.macd_top["macd"] > self.pre_macd_top["macd"] * self.macd_greater_threshold and self.macd_top["macd"] < self.macd_top["price"] * 0.015:
                    self.macd_greater_long= True
                else:
                    self.macd_greater_long= False

                if abs(self.macd_top["price"] - self.macd_top["sma"]) > abs(self.pre_macd_bottom["price"] - self.pre_macd_bottom["sma"]):
                    self.trend_side = "LONG"
                else: 
                    self.trend_side = "SKIP"

            # 最新的为空的macd
            if self.macd_top["index"] < self.macd_bottom["index"]:
                # === greater than the previous macd peak, reject extreme macd
                if self.macd_bottom["macd"] < self.pre_macd_bottom["macd"] * self.macd_greater_threshold and self.macd_bottom["macd"] > - self.macd_bottom["price"] * 0.015:
                    self.macd_greater_short = True
                else:
                    self.macd_greater_short = False

                if abs(self.macd_bottom["price"] - self.macd_bottom["sma"]) > abs(self.pre_macd_top["price"] - self.pre_macd_top["sma"]):
                    self.trend_side = "SHORT"
                else: 
                    self.trend_side = "SKIP"

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

            # -------- 第一层 --------
            if self.macd_greater_long and self.trend_side == "LONG":
                self.Long_pre1 = True
                self.Short_pre1 = False

            elif self.macd_greater_short and self.trend_side == "SHORT":
                self.Short_pre1 = True
                self.Long_pre1 = False


            # -------- 第二层 RSI --------
            if self.Long_pre1:
                if self.df["rsi14"].iloc[-1] < self.entry_rsi:
                    self.Long_pre2 = True

            if self.Short_pre1:
                if self.df["rsi14"].iloc[-1] > 100 - self.entry_rsi:
                    self.Short_pre2 = True


            # ===== LONG 80% invalidation =====
            if self.Long_pre1:
                if self.df["close"].iloc[-1] <= self.macd_top["invalid_price"] or self.df["close"].iloc[-1] > self.macd_top["price"]:
                    self.Long_pre1 = False
                    self.Long_pre2 = False
                    self.macd_greater_long= False
                    self.trend_side = "SKIP"      # ⭐ 加这一行
                    self.pre_macd_top = self.macd_top 
                    self.macd_top = None

            # ===== SHORT 80% invalidation ===== 
            if self.Short_pre1:
                if self.df["close"].iloc[-1] >= self.macd_bottom["invalid_price"] or self.df["close"].iloc[-1] > self.macd_bottom["price"]:
                    self.Short_pre1 = False
                    self.Short_pre2 = False
                    self.macd_greater_short= False
                    self.trend_side = "SKIP"      # ⭐ 加这一行
                    self.pre_macd_bottom = self.macd_bottom 
                    self.macd_bottom = None


            # -------- 第三层 K线触发 --------
            # 连续2根 bullish
            if self.Long_pre2:
                if long_streak >= self.required_bull_count :
                    
                    self.entry = "LONG"
                    self.position = "LONG"
                    self.entry_price = self.df["close"].iloc[-1]
                    self.entry_index = i

                    # ===== 计算止盈止损 =====
                    swing_high = self.macd_top["price"]
                    start_price = self.macd_top["start_price"]
                    retracement = (swing_high - self.entry_price) / (swing_high - start_price)

                    if retracement <= 0.5 and self.entry_price > self.df[f"ema{self.ema_window}"].iloc[-1] + (swing_high - start_price) * 0.1: 
                        self.long_sl = self.df[f"ema{self.ema_window}"].iloc[-1]
                        self.long_tp = swing_high + (swing_high - start_price) * self.tp_ratio
                    else:
                        self.long_sl = swing_high - (swing_high - start_price) * self.sl_ratio
                        self.long_tp = swing_high
                    

                    print(self.df.index[-1], "🚀 LONG ENTRY", self.df["close"].iloc[-1], "| TP: ",self.long_tp,"| SL: ",self.long_sl,"| Retracement: ",round(retracement,2))

                    # reset
                    self.Long_pre1 = False
                    self.Long_pre2 = False
                    self.macd_greater_long= False


            # 连续2根 bearish
            if self.Short_pre2:
                if short_streak >= self.required_bear_count:
                    
                    self.entry = "SHORT"
                    self.position = "SHORT"
                    self.entry_price = self.df["close"].iloc[-1]
                    self.entry_index = i

                    # 计算 SHORT 止盈止损
                    swing_low = self.macd_bottom["price"]
                    start_price = self.macd_bottom["start_price"]
                    retracement = (self.entry_price - swing_low) / (start_price - swing_low)
                    
                    if retracement <= 0.5 and self.entry_price < self.df[f"ema{self.ema_window}"].iloc[-1] - (swing_high - start_price) * 0.1:
                        self.short_sl = self.df[f"ema{self.ema_window}"].iloc[-1]
                        self.short_tp = swing_low + (swing_low - start_price) * self.tp_ratio
                    else:
                        self.short_sl = swing_low - (swing_low - start_price) * self.sl_ratio
                        self.short_tp = swing_low

                    

                    print(self.df.index[-1], "🔻 SHORT ENTRY", self.df["close"].iloc[-1], "| TP: ",self.short_tp,"| SL: ",self.short_sl,"| Retracement: ",round(retracement,2))
                    # reset
                    self.Short_pre1 = False
                    self.Short_pre2 = False
                    self.macd_greater_short = False
            

        # ===================
        # 持仓 / 出场逻辑
        # ===================
        if self.entry == "LONG":
            close = self.df["close"].iloc[-1]
            tp_hit = close >= self.long_tp
            sl_hit = close <= self.long_sl
            rsi_exit = self.df["rsi14"].iloc[-1] > self.exit_rsi

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

                print(self.df.index[-1], "❌ LONG EXIT", exit_price,
                    "| PnL:", round(pnl_pct,2), "%", "| Reason:", exit_reason)
                print("CUM:", round(self.cummulative_profit,2), "%\n")

                self.entry = False
                self.position = None

        if self.entry == "SHORT":
            close = self.df["close"].iloc[-1]
            tp_hit = close <= self.short_tp
            sl_hit = close >= self.short_sl
            rsi_exit = self.df["rsi14"].iloc[-1] < 100 - self.exit_rsi

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

                print(self.df.index[-1], "❌ SHORT EXIT", exit_price,
                    "| PnL:", round(pnl_pct,2), "%", "| Reason:", exit_reason)
                print("CUM:", round(self.cummulative_profit,2), "%\n")

                self.entry = False
                self.position = None


        print(
            f'| Greater: {self.macd_greater_long} {self.macd_greater_short} '
            f'| Trend: {self.trend_side} '
            f'| Long_K: {long_streak} '
            f'| Short_K: {short_streak} '
            f'| RSI: {round(self.df["rsi14"].iloc[-1],1)} '
            F'| Status: Pre_1 = {self.Long_pre1 or self.Short_pre1} & Pre_2 = {self.Long_pre2 or self.Short_pre2}'
        )


bitcoin = Bitcoin()

start_time = "2025-01-01 00:00:00"
end_time   = "2026-02-26 10:00:00"
symbol     = "BTCUSDT"

while True:
    try:
        bitcoin.get_latest_klines()
        #bitcoin.add_ema(8)
        #bitcoin.add_ema(26)
        bitcoin.add_ema()
        bitcoin.add_sma()
        bitcoin.add_rsi(14)
        bitcoin.add_macd()

        bitcoin.Strategy()

        time.sleep(30)
    except KeyboardInterrupt:
        break
    except Exception as e:
        print("ERROR:", e)
        time.sleep(30)


