import pandas as pd
import mplfinance as mpf
import datetime
from binance.client import Client
import time
import ta  
import numpy as np
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


# ========================
# Binance Client
# ========================


class Bitcoin():
    def __init__(self):
        self.df = pd.DataFrame()
        #====== parameter ======
        self.macd_greater_threshold = 1.1
        self.macd_window = 30  #  macd peak 的 window
        self.sma_window = 200   # 判断 macd long 还是 short 的 window
        self.ema_window = 120  # 0.5 以上的 止损线
        self.entry_rsi = 40   # 入场rsi
        self.exit_rsi = 70
        self.tp_ratio = 0.1     # 0 是前高， 0.1是 比前高还多10%
        self.sl_ratio = 0.8    # 1 是上涨起点 0.8 是小过起点
        self.invalid_drawback = 0.8
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

    def add_volume_ma(self, window):
        self.df[f'vol_ma'] = self.df['volume'].rolling(window).mean()

    # ==================
    # Get Klines
    # ==================
    def load_csv(self, filename):
        """
        直接读取 CSV，设置时间索引
        """
        return pd.read_csv(filename, index_col=0, parse_dates=True)

    def count_recent_bullish(self,i):

        close = self.df["close"].values
        open_ = self.df["open"].values

        count = 0

        while i >= 0 and close[i] > open_[i]:
            count += 1
            i -= 1

        return count


    def count_recent_bearish(self,i):

        close = self.df["close"].values
        open_ = self.df["open"].values

        count = 0

        while i >= 0 and close[i] < open_[i]:
            count += 1
            i -= 1

        return count
    
    # ==================================================
    # 统计重置
    # ==================================================
    def reset_stats(self):
        self.cummulative_profit = 0
        self.tp_count = 0
        self.sl_count = 0
        self.entry = False
# ==================
# Strategy
# ==================
    def Strategy(self):

        # ========= initialize ==========
        pre_macd_top = {
            "i" :0,
            "index" : None,
            "macd" :99999,
            "price" : 999999,
            "sma" :  999999,
        } 

        pre_macd_bottom = {
            "i" : 0,
            "index" : None,
            "macd" :-99999,
            "price" : -999999,
            "sma":  -999999,
        } 
        macd_greater = False
        trend_side = "SKIP"
        Long_pre1 = False
        Long_pre2 = False
        Short_pre1 = False
        Short_pre2 = False
        entry_price = 0
        entry_index = 0
        position = None  # LONG / SHORT
        long_invalid_price = None
        short_invalid_price = None
        long_tp = None
        long_sl = None
        short_tp = None
        short_sl = None
        #==========================================

        # ===== BACKTEST ==========
        for i in range(len(self.df)):


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
                    long_invalid_price = macd_top["price"] - swing_size * self.invalid_drawback
                else:
                    long_invalid_price = None

                # === greater than the previous macd peak , and reject extreme macd
                if macd_top["macd"] > pre_macd_top["macd"] * self.macd_greater_threshold and macd_top["macd"] < macd_top["price"] * 0.015:
                    macd_greater = True
                else:
                    macd_greater = False

                pre_macd_top = macd_top

                if abs(macd_top["price"] - macd_top["sma"]) > abs(pre_macd_bottom["price"] - pre_macd_bottom["sma"]):
                    trend_side = "LONG"
                else: 
                    trend_side = "SKIP"

                #print("Top   : ",self.df.index[i], self.df["macd_top"].iloc[i], self.df["macd_top_price"].iloc[i],macd_greater,trend_side)

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
                    short_invalid_price = macd_bottom["price"] + swing_size * self.invalid_drawback
                else:
                    short_invalid_price = None

                
                # === greater than the previous macd peak, reject extreme macd
                if macd_bottom["macd"] < pre_macd_bottom["macd"] * self.macd_greater_threshold and macd_bottom["macd"] > -macd_bottom["price"] * 0.015:
                    macd_greater = True
                else:
                    macd_greater = False

                pre_macd_bottom = macd_bottom

                if abs(macd_bottom["price"] - macd_bottom["sma"]) > abs(pre_macd_top["price"] - pre_macd_top["sma"]):
                    trend_side = "SHORT"
                else: 
                    trend_side = "SKIP"

                #print("Bottom: ",self.df.index[i], self.df["macd_bottom"].iloc[i], self.df["macd_bottom_price"].iloc[i],macd_greater,trend_side)


            bull_count = self.count_recent_bullish(i)
            bear_count = self.count_recent_bearish(i)


            # ===================
            # 入场逻辑
            # ===================
            if not self.entry:
                # ===== LONG 80% invalidation =====
                if Long_pre1 and long_invalid_price is not None:
                    if self.df["close"].iloc[i] <= long_invalid_price:
                        Long_pre1 = False
                        Long_pre2 = False
                        macd_greater = False
                        trend_side = "SKIP"      # ⭐ 加这一行
                # ===== SHORT 80% invalidation =====
                if Short_pre1 and short_invalid_price is not None:
                    if self.df["close"].iloc[i] >= short_invalid_price:
                        Short_pre1 = False
                        Short_pre2 = False
                        macd_greater = False
                        trend_side = "SKIP"      # ⭐ 加这一行

                # -------- 第一层 --------
                if macd_greater and trend_side == "LONG":
                    Long_pre1 = True
                    Short_pre1 = False

                elif macd_greater and trend_side == "SHORT":
                    Short_pre1 = True
                    Long_pre1 = False


                # -------- 第二层 RSI --------
                if Long_pre1:
                    if self.df["rsi14"].iloc[i] < self.entry_rsi:
                        Long_pre2 = True

                if Short_pre1:
                    if self.df["rsi14"].iloc[i] > 100 - self.entry_rsi:
                        Short_pre2 = True


                # -------- 第三层 K线触发 --------
                # 连续2根 bullish
                if Long_pre2:
                    if bull_count >= self.required_bull_count:
                        
                        self.entry = "LONG"
                        position = "LONG"
                        entry_price = self.df["close"].iloc[i]
                        entry_index = i

                        # ===== 计算止盈止损 =====
                        swing_high = pre_macd_top["price"]
                        start_price = pre_macd_top["start_price"]
                        retracement = (swing_high - entry_price) / (swing_high - start_price)

                        if retracement <= 0.5 and entry_price > self.df[f"ema{self.ema_window}"].iloc[i]: 
                            long_sl = self.df[f"ema{self.ema_window}"].iloc[i]
                            long_tp = swing_high + (swing_high - start_price) * self.tp_ratio
                        else:
                            long_sl = swing_high - (swing_high - start_price) * self.sl_ratio
                            long_tp = swing_high
                        

                        #print(self.df.index[i], "🚀 LONG ENTRY", self.df["close"].iloc[i], "| TP: ",long_tp,"| SL: ",long_sl,"| Retracement: ",round(retracement,2))

                        # reset
                        Long_pre1 = False
                        Long_pre2 = False
                        macd_greater = False


                # 连续2根 bearish
                if Short_pre2:
                    if bear_count >= self.required_bear_count:
                        
                        self.entry = "SHORT"
                        position = "SHORT"
                        entry_price = self.df["close"].iloc[i]
                        entry_index = i

                        # 计算 SHORT 止盈止损
                        swing_low = pre_macd_bottom["price"]
                        start_price = pre_macd_bottom["start_price"]
                        retracement = (entry_price - swing_low) / (start_price - swing_low)
                        
                        if retracement <= 0.5 and entry_price < self.df[f"ema{self.ema_window}"].iloc[i]:
                            short_sl = self.df[f"ema{self.ema_window}"].iloc[i]
                            short_tp = swing_low + (swing_low - start_price) * self.tp_ratio
                        else:
                            short_sl = swing_low - (swing_low - start_price) * self.sl_ratio
                            short_tp = swing_low

                        

                        #print(self.df.index[i], "🔻 SHORT ENTRY", self.df["close"].iloc[i], "| TP: ",short_tp,"| SL: ",short_sl,"| Retracement: ",round(retracement,2))
                        # reset
                        Short_pre1 = False
                        Short_pre2 = False
                        macd_greater = False
                

            # ===================
            # 持仓 / 出场逻辑
            # ===================
            if self.entry == "LONG":
                close = self.df["close"].iloc[i]
                tp_hit = close >= long_tp
                sl_hit = close <= long_sl
                rsi_exit = self.df["rsi14"].iloc[i] > self.exit_rsi

                if tp_hit or sl_hit or rsi_exit:
                    exit_price = close
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
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

                    #print(self.df.index[i], "❌ LONG EXIT", exit_price,
                    #    "| PnL:", round(pnl_pct,2), "%", "| Reason:", exit_reason)
                    #print("CUM:", round(self.cummulative_profit,2), "%\n")

                    self.entry = False
                    position = None

            if self.entry == "SHORT":
                close = self.df["close"].iloc[i]
                tp_hit = close <= short_tp
                sl_hit = close >= short_sl
                rsi_exit = self.df["rsi14"].iloc[i] < 100 - self.exit_rsi

                if tp_hit or sl_hit or rsi_exit:
                    exit_price = close
                    pnl_pct = (entry_price - exit_price) / entry_price * 100
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

                    #print(self.df.index[i], "❌ SHORT EXIT", exit_price,
                    #    "| PnL:", round(pnl_pct,2), "%", "| Reason:", exit_reason)
                    #print("CUM:", round(self.cummulative_profit,2), "%\n")

                    self.entry = False
                    position = None

        # ================= RESULT =================
        total_trades = self.tp_count + self.sl_count

        if total_trades == 0:
            return None

        return {
            "total_profit": self.cummulative_profit,
            "trades": total_trades,
            "win_rate": self.tp_count / total_trades,
            "tp": self.tp_count,
            "sl": self.sl_count
        }


def run_one_param(args):
    params, base_df = args

    bot = Bitcoin()
    bot.df = base_df.copy()

    # ===== 注入参数 =====
    for k, v in params.items():
        setattr(bot, k, v)

    bot.add_ema()
    bot.add_sma()
    bot.reset_stats()

    stats = bot.Strategy()

    if stats:
        return {**params, **stats}
    return None

if __name__ == "__main__":

    # ========= 参数组合 =========
    PARAM_GRID = {
        "macd_greater_threshold": [1.05, 1.1, 1.2],
        "macd_window": [20, 25, 30, 35],
        "sma_window": [100, 150, 200],
        "ema_window": [60, 90, 120],
        "entry_rsi": [30, 35, 40],
        "exit_rsi": [65, 70, 75],
        "tp_ratio": [0, 0.05, 0.1],
        "sl_ratio": [0.6, 0.8],
        "tp_ratio": [0.1, 0.2, 0.3],
        "sl_ratio": [0.7, 0.8, 0.9],
        "invalid_drawback": [0.7, 0.8],
        "required_bull_count": [2, 3, 4],
        "required_bear_count": [2, 3, 4]
    }
    '''
        self.macd_greater_threshold = 1.1
        self.macd_window = 30  #  macd peak 的 window
        self.sma_window = 200   # 判断 macd long 还是 short 的 window
        self.ema_window = 120  # 0.5 以上的 止损线
        self.entry_rsi = 40   # 入场rsi
        self.exit_rsi = 70
        self.tp_ratio = 0.1     # 0 是前高， 0.1是 比前高还多10%
        self.sl_ratio = 0.8    # 1 是上涨起点 0.8 是小过起点
        self.invalid_drawback = 0.8
        self.required_bull_count = 3
        self.required_bear_count = 3
    '''
    param_list = list(itertools.product(*PARAM_GRID.values()))
    param_dicts = [dict(zip(PARAM_GRID.keys(), v)) for v in param_list]

    print(f"Total param sets: {len(param_dicts)}")

    # ========= K线只拉一次 =========
    base_bot = Bitcoin()
    base_bot.df = base_bot.load_csv("BTCUSDT_15m_2025-12-01_2026-02-26.csv")

    base_bot.add_ema()
    base_bot.add_sma()
    base_bot.add_rsi(14)
    base_bot.add_macd()
    base_bot.add_macd_peak()
    base_bot.add_volume_ma(10)

    base_df = base_bot.df

    # ========= multiprocessing =========
    tasks = [(p, base_df) for p in param_dicts]
    results = []
    print(cpu_count())

    with Pool(20) as pool:
        for r in tqdm(
            pool.imap_unordered(run_one_param, tasks),
            total=len(tasks),
            desc="🚀 Optimizing"
        ):
            if r:
                results.append(r)

    # ========= 排序 =========
    top20 = sorted(results, key=lambda x: x["total_profit"], reverse=True)[:20]

    print("\n========= TOP 20 =========")
    for i, r in enumerate(top20, 1):
        print(f"\n🏆 Rank {i}")
        print(f"Params: { {k:r[k] for k in PARAM_GRID.keys()} }")
        print(
            f"Trades={r['trades']} | "
            f"WinRate={r['win_rate']*100:.1f}% | "
            f"Total={r['total_profit']:.2f}%"
        )
