#!/usr/bin/python
#!/usr/bin/python
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
import backtrader as bt
import matplotlib.pyplot as plt
import datetime

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 设置绘图参数
plt.rcParams['figure.figsize'] = [14, 10]
plt.rcParams['font.size'] = 12


class DataProcessor:
    def __init__(self, data_main, data_hedge, window=480, mtm_threshold=(0.8, 0.15), std_threshold=0.8, vol_drift=0.8):
        self.data_main = data_main
        self.data_hedge = data_hedge
        self.window = window
        self.mtm_upper, self.mtm_lower = mtm_threshold
        self.std_threshold=std_threshold
        self.vol_drift = vol_drift
        self.main = None
        self.hedge = None
        self.target_cols = ["date", "open", "high", "low", "close", "volume"]

    @staticmethod
    def _rolling_pct(win):
        if len(win) < 1:
            return np.nan
        return percentileofscore(win, win[-1]) / 100

    def _cal_rank(self, series):
        return series.rolling(
            window=self.window,
            min_periods=1
        ).apply(self._rolling_pct, raw=True)

    def _load_data(self):
        self.main = pd.read_csv(self.data_main, usecols=self.target_cols, parse_dates=["date"])
        self.hedge = pd.read_csv(self.data_hedge, usecols=self.target_cols, parse_dates=["date"])

    def _preprocess(self):
        self.main["date"] = pd.to_datetime(self.main["date"])
        self.hedge["date"] = pd.to_datetime(self.hedge["date"])

    def _cal_features(self):
        # ===== 处理主资产 =====
        self.main["return"] = self.main["close"].pct_change(20)
        self.main["std_return"] = self.main["return"].rolling(60).std()
        self.main["vol_ma20"] = self.main["volume"].rolling(20).mean()
        self.main["vol_drift"] = (self.main["volume"] / self.main["vol_ma20"]) - 1
        self.main.dropna(inplace=True)

        # 拥挤度因子
        self.main["std_rank"] = self._cal_rank(self.main["std_return"])

        # ===== 处理对冲资产 =====
        self.hedge["return"] = self.hedge["close"].pct_change(20)
        self.hedge["std_return"] = self.hedge["return"].rolling(60).std()
        self.hedge["vol_ma20"] = self.hedge["volume"].rolling(20).mean()
        self.hedge["vol_drift"] = (self.hedge["volume"] / self.main["vol_ma20"]) - 1
        self.hedge.dropna(inplace=True)
        # 拥挤度因子
        self.hedge["std_rank"] = self._cal_rank(self.hedge["std_return"])

        # ===== 动量因子 =====
        self.main["mtm"] = self.main["return"] - self.hedge["return"]
        self.main["pct_rank"] = self._cal_rank(self.main["mtm"])


    def _generate_signal(self):
        self.main["mtm_signal"] = np.select(
            [
                self.main["pct_rank"] >= self.mtm_upper,
                self.main["pct_rank"] <= self.mtm_lower
            ],
            [1,-1],
            default=0
        )

        self.main["std_signal"] = np.select(
            [
                (self.hedge["std_rank"] >= self.std_threshold) & (self.main["std_rank"] < self.std_threshold),
                (self.main["std_rank"] >= self.std_threshold) & (self.hedge["std_rank"] < self.std_threshold),
            ],
            [1,-1],
            default=0
        )

        self.main["vol_signal"] = np.select(
            [
                (self.main["vol_drift"] >= self.vol_drift) & (self.main["return"] > 0),
                (self.main["vol_drift"] >= self.vol_drift) & (self.main["return"] < 0),
            ],
            [-1, 1],
            default=0
        )

        self.hedge["vol_signal"] = np.select(
            [
                (self.hedge["vol_drift"] >= self.vol_drift) & (self.hedge["return"] > 0),
                (self.hedge["vol_drift"] >= self.vol_drift) & (self.hedge["return"] < 0),
            ],
            [-1, 1],
            default=0
        )


        self._clean_data()

    def _clean_data(self):
        main_target = [
            "date", "open", "high",
            "low", "close", "volume",
            "std_return", "mtm_signal", "std_signal", "vol_signal"
        ]
        hedge_target = [
            "date", "open", "high",
            "low", "close", "volume",
            "std_return", "vol_signal"
        ]
        self.main = self.main[main_target]
        self.hedge = self.hedge[hedge_target]
        self.main.dropna(inplace=True)
        self.hedge.dropna(inplace=True)

    def save_data(self, main_path, hedge_path):
        if self.main is not None:
            self.main.to_csv(main_path, index=False)
        else:
            raise ValueError("Execute process()")
        if self.hedge is not None:
            self.hedge.to_csv(hedge_path, index=False)
        else:
            raise ValueError("Execute process()")

    def process(self):
        self._load_data()
        self._preprocess()
        self._cal_features()
        self._generate_signal()

        return self.main, self.hedge


class ValueRecorder(bt.Analyzer):
    def start(self):
        self.values = []

    def next(self):
        self.values.append(self.strategy.broker.getvalue())

# 交易策略（优化执行逻辑和绩效统计）
class MyStrategy(bt.Strategy):
    params = (
        ("trade_mode", "SHORT_BIG"),
        ("take_profit", 0.15),
        ("stop_loss", 0.05),
        ("max_holding_days", 10),
    )
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt}: {txt}")

    def __init__(self):
        self.trade_log = []

        self.data_hedge = self.datas[0]
        self.data_main = self.datas[1]

        self.entry_price_hedge = None
        self.entry_price_main = None
        self.entry_date_hedge = None
        self.entry_date_main = None      # 记录买入价和买入日期

        self.position_return_hedge = []   # 记录持有收益
        self.position_return_main = []

        self.last_trade_date = None     # 记录上一次交易日期

        # 创建不同数据源的交易信号记录器
        bt.observers.BuySell(self.datas[0])
        bt.observers.BuySell(self.datas[1])

        self.total_trades = 0
        self.daily_values = []  # 记录每日净值
        self.win_trades = 0

        self.pnl_list = []

        self.entry_bar_main = None
        self.entry_bar_hedge = None

    def _log_trade(self, action ,symbol, price, size, reason):
        """记录交易日志"""
        total_value = self.broker.getvalue()
        trade_info = {
            "Date": self.datetime.date(0),
            "Action": action,
            "Symbol": symbol,
            "Price": price,
            "Size": size,
            "Reason": reason,
            "total_value": total_value,
        }
        self.trade_log.append(trade_info)
        print(f"{self.datetime.date(0)} | {action} | {symbol} | Price: {price:.2f} | Size: {size:.2f} | Reason: {reason}")

    def notify_trade(self, trade):
        """记录交易盈亏"""
        if not trade.isclosed:
            return
        if trade.isclosed:
            pnl = trade.pnlcomm
            self.pnl_list.append(pnl)
            self.total_trades += 1
            self.win_trades += (pnl>0)

    def _resolve_signals(self, mtm, std):
        signal_strength = {
            "long_small": 1,
            "long_big": -1,
            "neutral": 0
        }
        if std != 0:
            return std
        return mtm

    def _cal_position(self, price_hedge, price_main, vol_hedge, vol_main, lot_size=1, cash_buffer=0.05):
        """
        优化后的仓位计算逻辑
        """
        total_value = self.broker.getvalue()
        available_cash = self.broker.get_cash() * (1 - cash_buffer)  # 合并现金缓冲计算
        target_allocation = min(total_value * 0.5, available_cash * 0.5)

        # --- 仓位计算函数 ---
        def calc_lots(target_value, price, vol):
            if price <= 0:
                return 0
            raw_lots = target_value / price
            adjusted_lots = max(int(raw_lots // lot_size) * lot_size, lot_size)

            # 根据成交量调整仓位
            if vol == 1:
                adjusted_lots = int(adjusted_lots * 1.5)  # 增加
            elif vol == -1:
                adjusted_lots = int(adjusted_lots * 0.9)  # 减少

            # 确保不超过最大可用资金
            return max(lot_size, min(adjusted_lots, int(available_cash / price // lot_size) * lot_size))
        if self.p.trade_mode == "LONG":
            position_hedge = calc_lots(target_allocation, price_hedge, vol_hedge)
            position_main = calc_lots(target_allocation, price_main, vol_main)
        else:
            # 计算初步仓位
            position_hedge = calc_lots(target_allocation * 0.5, price_hedge, vol_hedge)
            position_main = calc_lots(target_allocation * 0.5, price_main, vol_main)

        # # --- 确保总成本不超过 target_allocation ---
        # total_cost = position_hedge * price_hedge + position_main * price_main
        # if total_cost > target_allocation:
        #     scale_factor = target_allocation / total_cost  # 计算缩减比例
        #     position_hedge = max(lot_size, int(position_hedge * scale_factor // lot_size) * lot_size)
        #     position_main = max(lot_size, int(position_main * scale_factor // lot_size) * lot_size)

        return position_hedge, position_main


    def _execute_trading(self, signal):
        # 当前时间
        current_date = self.datetime.date(0)

        pos_hedge = self.getposition(self.data_hedge).size
        pos_main = self.getposition(self.data_main).size

        price_hedge = self.data_hedge.close[0]
        price_main = self.data_main.close[0]

        vol_hedge = self.data_hedge.vol_signal[0]
        vol_main = self.data_main.vol_signal[0]


        """===== 执行交易逻辑 ====="""
        if self.p.trade_mode == "SHORT_BIG":
            size_hedge, size_main = self._cal_position(price_hedge, price_main, vol_hedge, vol_main)
            if signal == 1:  # 做多 main，做空 hedge
                if pos_main == 0:  # 无 main 持仓
                    self.buy(self.data_main, size=size_main, exectype=bt.Order.Market)
                    self._log_trade("BUY", "SMALL", price_main, size_main, "LONG SMALL")
                    self.entry_price_main, self.entry_bar_main = price_main, len(self.data_main)

                if pos_hedge >= 0:  # 只有当 hedge 无空头时才需要调整
                    if pos_hedge > 0:
                        self.close(self.data_hedge)
                        self._log_trade("CLOSE", "BIG", price_hedge, pos_hedge, "SHORT BIG")
                    self.sell(self.data_hedge, size=size_hedge, exectype=bt.Order.Market)
                    self._log_trade("SELL", "BIG", price_hedge, size_hedge, "SHORT BIG")
                    self.entry_price_hedge, self.entry_bar_hedge = price_hedge, len(self.data_hedge)

            elif signal == -1:  # 做多 hedge，平掉 main
                if pos_main > 0:
                    self.close(self.data_main)
                    self._log_trade("CLOSE", "SMALL", price_main, pos_main, "LONG BIG")

                if pos_hedge <= 0:  # 只有当 hedge 无多头时才需要调整
                    if pos_hedge < 0:
                        self.close(self.data_hedge)
                        self._log_trade("CLOSE", "BIG", price_hedge, pos_hedge, "LONG BIG")
                    self.buy(self.data_hedge, size=size_hedge, exectype=bt.Order.Market)
                    self._log_trade("BUY", "BIG", price_hedge, size_hedge, "LONG BIG")
                    self.entry_price_hedge, self.entry_bar_hedge = price_hedge, len(self.data_hedge)

        if self.p.trade_mode == "LONG_SHORT":
            size_hedge, size_main = self._cal_position(price_hedge, price_main, vol_hedge, vol_main)
            if signal == 1:  # 做多 main，做空 hedge
                if pos_main <= 0:
                    if pos_main < 0:
                        self.close(self.data_hedge)
                        self._log_trade("CLOSE", "SMALL", price_main, pos_main, "LONG SMALL")
                    self.buy(self.data_main, size=size_main, exectype=bt.Order.Market)
                    self._log_trade("BUY", "SMALL", price_main, size_main, "LONG SMALL")
                    self.entry_price_main, self.entry_bar_main = price_main, len(self.data_main)

                if pos_hedge >= 0:  # 只有当 hedge 无空头时才需要调整
                    if pos_hedge > 0:
                        self.close(self.data_hedge)
                        self._log_trade("CLOSE", "BIG", price_hedge, pos_hedge, "SHORT BIG")
                    self.sell(self.data_hedge, size=size_hedge, exectype=bt.Order.Market)
                    self._log_trade("SELL", "BIG", price_hedge, size_hedge, "SHORT BIG")
                    self.entry_price_hedge, self.entry_bar_hedge = price_hedge, len(self.data_hedge)

            elif signal == -1:  # 做多 hedge，做空 main
                if pos_main >= 0:
                    if pos_main > 0:
                        self.close(self.data_main)
                        self._log_trade("CLOSE", "SMALL", price_main, pos_main, "SHORT SMALL")
                self.sell(self.data_main, size=size_main, exectype=bt.Order.Market)
                self._log_trade("SELL", "SMALL", price_main, size_main, "SHORT SMALL")
                self.entry_price_main, self.entry_bar_main = price_main, len(self.data_main)

                if pos_hedge <= 0:  # 只有当 hedge 无多头时才需要调整
                    if pos_hedge < 0:
                        self.close(self.data_hedge)
                        self._log_trade("CLOSE", "BIG", price_hedge, pos_hedge, "LONG BIG")
                    self.buy(self.data_hedge, size=size_hedge, exectype=bt.Order.Market)
                    self._log_trade("BUY", "BIG", price_hedge, size_hedge, "LONG BIG")
                    self.entry_price_hedge, self.entry_bar_hedge = price_hedge, len(self.data_hedge)

        if self.p.trade_mode == "SHORT_CLOSE":
            size_hedge, size_main = self._cal_position(price_hedge, price_main, vol_hedge, vol_main)
            if signal == 1:  # 做多 main，做空 hedge
                if pos_main == 0:  # 无 main 持仓
                    self.buy(self.data_main, size=size_main, exectype=bt.Order.Market)
                    self._log_trade("BUY", "SMALL", price_main, size_main, "LONG SMALL")
                    self.entry_price_main, self.entry_bar_main = price_main, len(self.data_main)

                if pos_hedge >= 0:  # 只有当 hedge 无空头时才需要调整
                    if pos_hedge > 0:
                        self.close(self.data_hedge)
                        self._log_trade("CLOSE", "BIG", price_hedge, pos_hedge, "SHORT BIG")
                    self.sell(self.data_hedge, size=size_hedge, exectype=bt.Order.Market)
                    self._log_trade("SELL", "BIG", price_hedge, size_hedge, "SHORT BIG")
                    self.entry_price_hedge, self.entry_bar_hedge = price_hedge, len(self.data_hedge)

            elif signal == -1:  # 做多 hedge，平掉 main
                if pos_main > 0:
                    self.close(self.data_main)
                    self._log_trade("CLOSE", "SMALL", price_main, pos_main, "CLOSE SMALL")

                if pos_hedge <= 0:  # 只有当 hedge 无多头时才需要调整
                    if pos_hedge < 0:
                        self.close(self.data_hedge)
                        self._log_trade("CLOSE", "BIG", price_hedge, pos_hedge, "LONG BIG")

        if self.p.trade_mode == "LONG":
            if signal == 1:  # 做多 main，
                if pos_main == 0:  # 无 main 持仓
                    size_hedge, size_main = self._cal_position(price_hedge, price_main, vol_hedge, vol_main)
                    self.buy(self.data_main, size=size_main, exectype=bt.Order.Market)
                    self._log_trade("BUY", "SMALL", price_main, size_main, "LONG SMALL")
                    self.entry_price_main, self.entry_bar_main = price_main, len(self.data_main)

                if pos_hedge > 0:  # 只有当 hedge 无空头时才需要调整
                    self.close(self.data_hedge)
                    self._log_trade("CLOSE", "BIG", price_hedge, pos_hedge, "CLOSE BIG")

            elif signal == -1:  # 做多 hedge，平掉 main
                if pos_main > 0:
                    self.close(self.data_main)
                    self._log_trade("CLOSE", "SMALL", price_main, pos_main, "CLOSE SMALL")

                if pos_hedge == 0:  # 只有当 hedge 无多头时才需要调整
                    size_hedge, size_main = self._cal_position(price_hedge, price_main, vol_hedge, vol_main)
                    self.buy(self.data_hedge, size=size_hedge, exectype=bt.Order.Market)
                    self._log_trade("BUY", "BIG", price_hedge, size_hedge, "LONG BIG")
                    self.entry_price_hedge, self.entry_bar_hedge = price_hedge, len(self.data_hedge)


    def next(self):
        current_date = self.datetime.date(0)
        # 记录每日净值
        self.daily_values.append(self.broker.getvalue())

        # 止盈止损检查
        if self._check_take_profit_stop_loss():
            return

        # 10日无触发止盈止损，直接平仓
        if self._check_max_holding_days():
            return

        # 以上都没有，正常执行交易
        if self.last_trade_date and (current_date - self.last_trade_date).days < 5:
            return
        mtm_signal = self.data_main.mtm_signal[0]
        std_signal = self.data_main.std_signal[0]
        signal = self._resolve_signals(mtm_signal, std_signal)
        self._execute_trading(signal)
        self.last_trade_date = current_date



    def _check_take_profit_stop_loss(self):
        """检查止盈止损条件"""
        trigger = False
        # 今日持仓
        pos_hedge = self.getposition(self.data_hedge).size
        pos_main= self.getposition(self.data_main).size
        # 今日价格
        price_hedge = self.data_hedge.close[0]
        price_main = self.data_main.close[0]

        if self.p.trade_mode == "SHORT_BIG":
            # 检查持仓
            if pos_hedge != 0  and self.entry_price_hedge:
                if pos_hedge > 0: # 多300的情况
                    tp = self.entry_price_hedge * (1+self.p.take_profit)
                    sl = self.entry_price_hedge * (1-self.p.stop_loss)
                    if price_hedge >= tp or price_hedge <= sl:
                        reason = "Take Profit" if price_hedge >= tp else "Stop Loss"
                        self.close(self.data_hedge)
                        self._log_trade("CLOSE", "BIG", price_hedge, pos_hedge, f"{reason} triggered.")
                        self._reset_position("hedge")
                        trigger = True
                else:    # 空300的情况
                    tp = self.entry_price_hedge * (1-self.p.take_profit)
                    sl = self.entry_price_hedge * (1+self.p.stop_loss)
                    if price_hedge <= tp or price_hedge >= sl:
                        reason = "Take Profit" if price_hedge <= tp else "Stop Loss"
                        self.close(self.data_hedge)
                        self._log_trade("CLOSE", "BIG", price_hedge, abs(pos_hedge), f"{reason} triggered.")
                        self._reset_position("hedge")
                        trigger = True

            if pos_main > 0 and self.entry_price_main:
                tp = self.entry_price_main * (1 + self.p.take_profit)
                sl = self.entry_price_main * (1 - self.p.stop_loss)
                if price_main >= tp or price_main <= sl:
                    reason = "Take Profit" if price_main >= tp else "Stop Loss"
                    self.close(self.data_main)
                    self._log_trade("CLOSE", "SMALL", price_main, abs(pos_main), f"{reason} triggered.")
                    self._reset_position("main")
                    trigger = True

        if self.p.trade_mode == "LONG_SHORT":
            # 检查持仓
            if pos_hedge != 0 and self.entry_price_hedge:
                if pos_hedge > 0:  # 多300的情况
                    tp = self.entry_price_hedge * (1 + self.p.take_profit)
                    sl = self.entry_price_hedge * (1 - self.p.stop_loss)
                    if price_hedge >= tp or price_hedge <= sl:
                        reason = "Take Profit" if price_hedge >= tp else "Stop Loss"
                        self.close(self.data_hedge)
                        self._log_trade("CLOSE", "BIG", price_hedge, pos_hedge, f"{reason} triggered.")
                        self._reset_position("hedge")
                        trigger = True
                else:  # 空300的情况
                    tp = self.entry_price_hedge * (1 - self.p.take_profit)
                    sl = self.entry_price_hedge * (1 + self.p.stop_loss)
                    if price_hedge <= tp or price_hedge >= sl:
                        reason = "Take Profit" if price_hedge <= tp else "Stop Loss"
                        self.close(self.data_hedge)
                        self._log_trade("CLOSE", "BIG", price_hedge, abs(pos_hedge), f"{reason} triggered.")
                        self._reset_position("hedge")
                        trigger = True

            if pos_main != 0 and self.entry_price_main:
                if pos_main > 0:
                    tp = self.entry_price_main * (1 + self.p.take_profit)
                    sl = self.entry_price_main * (1 - self.p.stop_loss)
                    if price_main >= tp or price_main <= sl:
                        reason = "Take Profit" if price_main >= tp else "Stop Loss"
                        self.close(self.data_main)
                        self._log_trade("CLOSE", "SMALL", price_main, abs(pos_main), f"{reason} triggered.")
                        self._reset_position("main")
                        trigger = True
                else:  # 空的情况
                    tp = self.entry_price_main * (1 - self.p.take_profit)
                    sl = self.entry_price_main * (1 + self.p.stop_loss)
                    if price_main <= tp or price_main >= sl:
                        reason = "Take Profit" if price_main <= tp else "Stop Loss"
                        self.close(self.data_main)
                        self._log_trade("CLOSE", "SMALL", price_main, abs(pos_main), f"{reason} triggered.")
                        self._reset_position("main")
                        trigger = True

        if self.p.trade_mode == "SHORT_CLOSE":
            # 检查持仓
            if pos_hedge != 0 and self.entry_price_hedge:
                if pos_hedge > 0:  # 多300的情况
                    tp = self.entry_price_hedge * (1 + self.p.take_profit)
                    sl = self.entry_price_hedge * (1 - self.p.stop_loss)
                    if price_hedge >= tp or price_hedge <= sl:
                        reason = "Take Profit" if price_hedge >= tp else "Stop Loss"
                        self.close(self.data_hedge)
                        self._log_trade("CLOSE", "BIG", price_hedge, pos_hedge, f"{reason} triggered.")
                        self._reset_position("hedge")
                        trigger = True
                else:  # 空300的情况
                    tp = self.entry_price_hedge * (1 - self.p.take_profit)
                    sl = self.entry_price_hedge * (1 + self.p.stop_loss)
                    if price_hedge <= tp or price_hedge >= sl:
                        reason = "Take Profit" if price_hedge <= tp else "Stop Loss"
                        self.close(self.data_hedge)
                        self._log_trade("CLOSE", "BIG", price_hedge, abs(pos_hedge), f"{reason} triggered.")
                        self._reset_position("hedge")
                        trigger = True

            if pos_main > 0 and self.entry_price_main:
                tp = self.entry_price_main * (1 + self.p.take_profit)
                sl = self.entry_price_main * (1 - self.p.stop_loss)
                if price_main >= tp or price_main <= sl:
                    reason = "Take Profit" if price_main >= tp else "Stop Loss"
                    self.close(self.data_main)
                    self._log_trade("CLOSE", "SMALL", price_main, abs(pos_main), f"{reason} triggered.")
                    self._reset_position("main")
                    trigger = True

        if self.p.trade_mode == "LONG":
            # 检查持仓
            if pos_hedge != 0 and self.entry_price_hedge:
                if pos_hedge > 0:  # 多300的情况
                    tp = self.entry_price_hedge * (1 + self.p.take_profit)
                    sl = self.entry_price_hedge * (1 - self.p.stop_loss)
                    if price_hedge >= tp or price_hedge <= sl:
                        reason = "Take Profit" if price_hedge >= tp else "Stop Loss"
                        self.close(self.data_hedge)
                        self._log_trade("CLOSE", "BIG", price_hedge, pos_hedge, f"{reason} triggered.")
                        self._reset_position("hedge")
                        trigger = True

            if pos_main > 0 and self.entry_price_main:
                tp = self.entry_price_main * (1 + self.p.take_profit)
                sl = self.entry_price_main * (1 - self.p.stop_loss)
                if price_main >= tp or price_main <= sl:
                    reason = "Take Profit" if price_main >= tp else "Stop Loss"
                    self.close(self.data_main)
                    self._log_trade("CLOSE", "SMALL", price_main, abs(pos_main), f"{reason} triggered.")
                    self._reset_position("main")
                    trigger = True

        return trigger

    def _check_max_holding_days(self):
        trigger = False
        """检查是否超过最大持仓天数"""
        current_date = self.datetime.date(0)
        # 今日持仓
        pos_hedge = self.getposition(self.data_hedge).size
        pos_main = self.getposition(self.data_main).size

        if self.entry_bar_hedge and pos_hedge != 0:
            holding_days = len(self.data_hedge) - self.entry_bar_hedge
            if holding_days >= self.p.max_holding_days:
                self.close(self.data_hedge)
                self._log_trade("CLOSE", "BIG", self.data_hedge.close[0], abs(pos_hedge), f"Forced CLOSE after {holding_days} days.")
                self._reset_position("hedge")
                trigger = True

        if self.entry_date_main and pos_main != 0:
            holding_days = len(self.data_main) - self.entry_bar_main
            if holding_days >= self.p.max_holding_days:
                self.close(self.data_main)
                self._log_trade("CLOSE", "SMALL", self.data_main.close[0], abs(pos_main),
                                f"Forced CLOSE after {holding_days} days.")
                self._reset_position("main")
                trigger = True

        return trigger

    def _reset_position(self, asset_type):
        # ===== 重置仓位记录 =====
        if asset_type == "hedge":
            self.entry_price_hedge = None
            self.entry_date_hedge = None
        if asset_type == "main":
            self.entry_price_main = None
            self.entry_date_main = None

    def save_trade_log(self):
        filename = f"{self.p.trade_mode}_trade_log.csv"
        df = pd.DataFrame(self.trade_log)
        df.to_csv(filename, index=False)
        self.log(f"Trade log saved to {filename}")

    def stop(self):
        self.save_trade_log()
        # 转换为 Pandas Series 以便计算
        self.daily_values = pd.Series(self.daily_values)
        returns = self.daily_values.pct_change().dropna()

        dates = pd.to_datetime([bt.num2date(d) for d in self.datas[0].datetime.array])
        returns.index = dates[1:]  # returns 少一个元素（由于 pct_change）

        # 计算年化收益率（CAGR）
        total_years = len(returns) / 252
        final_value = self.daily_values.iloc[-1]
        initial_value = self.daily_values.iloc[0]
        annualized_return = (final_value / initial_value) ** (1 / total_years) - 1

        # 计算夏普比率（避免 std 为零）
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan

        # 计算最大回撤
        peak = self.daily_values.expanding().max()
        drawdown = (peak - self.daily_values) / peak
        max_drawdown = drawdown.max()

        # 计算 Calmar Ratio（年化收益 / 最大回撤）
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else np.nan

        # 计算 Sortino Ratio（只计算下行波动率）
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino_ratio = returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else np.nan

        # 计算盈亏比
        wins = [x for x in self.pnl_list if x > 0]
        losses = [x for x in self.pnl_list if x < 0]
        profit_factor = abs(sum(wins) / sum(losses)) if losses else np.nan

        # **计算胜率**
        total_trades = len(self.pnl_list)
        winning_trades = len(wins)
        win_rate = winning_trades / total_trades if total_trades > 0 else np.nan

        # 计算最大连续亏损天数（优化）
        loss_streaks = (returns < 0).astype(int).groupby((returns >= 0).cumsum()).cumsum()
        max_consecutive_losses = loss_streaks.max()

        # 计算平均盈利和亏损（避免空列表时报错）
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        # **计算年度收益率**
        annual_returns = (1 + returns).resample("Y").prod() - 1
        annual_returns.index = annual_returns.index.year  # 只保留年份

        print("\n========== 回测结果 ==========")
        print(f"初始资金: 1,000,000.00 CNY")
        print(f"期末资金: {final_value:,.2f} CNY")
        print(f"年化收益率 (CAGR): {annualized_return:.2%}")
        print(f"夏普比率: {sharpe_ratio:.2f}")
        print(f"Sortino 比率: {sortino_ratio:.2f}")
        print(f"最大回撤: {max_drawdown:.2%}")
        print(f"Calmar 比率: {calmar_ratio:.2f}")
        print(f"总交易次数: {self.total_trades}")
        print(f"胜率 (Win Rate): {win_rate:.2%}")
        print(f"盈亏比 (Profit Factor): {profit_factor:.2f}")
        print(f"最大连续亏损天数: {max_consecutive_losses}")
        print(f"平均每笔盈利: {avg_win:.2f}")
        print(f"平均每笔亏损: {avg_loss:.2f}")
        # **打印年度收益率**
        print("\n========== 分年度收益 ==========")
        for year, ret in annual_returns.items():
            print(f"{year}: {ret:.2%}")


class ExtendData(bt.feeds.PandasData):
    lines = ("mtm_signal", "std_signal", "std_return", "vol_signal")
    params = (
        ("dtformat", "%Y-%m-%d"),
        ("fromdate", datetime.datetime(2016,1,1)),
        ("std_return", 6),
        ("mtm_signal", -1),
        ("std_signal", -1),
        ("vol_signal", -1),
        ("openinterest", -1),
    )

if __name__ == "__main__":
    processor = DataProcessor(
        data_main= r"",
        data_hedge=r"",
        window=750,
        mtm_threshold=(0.7, 0.15),
        std_threshold=0.8,
        vol_drift=0.9
    )

    main_df, hedge_df = processor.process()
    # processor.save_data("main_data.csv", "hedge_data.csv")


    data_hedge = ExtendData(
        dataname=hedge_df,
        name="BIG",
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        std_return=6,
        vol_signal=7,
        timeframe=bt.TimeFrame.Days,
        compression=1
    )

    data_main = ExtendData(
        dataname=main_df,
        name="SMALL",
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        std_return=6,
        mtm_signal=7,
        std_signal=8,
        vol_signal=9,
        timeframe=bt.TimeFrame.Days,
        compression=1
    )



    portfolio_values = {}

    """
    LONG_SHORT: 对应第一类，多空
    SHORT_CLOSE: 对应第二类，可以多小盘空大盘，触发开多大盘信号时全部平仓
    LONG: 对应第三类，纯多头
    SHORT_BIG: 对应第四类，可以多小盘空大盘，触发开多大盘信号时平仓小盘，开多大盘
    """

    for mode in ["LONG_SHORT", "SHORT_CLOSE", "LONG", "SHORT_BIG"]:
        cerebro = bt.Cerebro()
        cerebro.adddata(data_hedge)
        cerebro.adddata(data_main)

        cerebro.broker.set_cash(1000000)
        cerebro.broker.setcommission(commission=0.0003)  # 设置交易手续费
        print(">>>>>>>>>><<<<<<<<<<")
        print(f"===== {mode} 回测开始 =====")
        cerebro.addstrategy(
            MyStrategy,
            trade_mode=mode,
            take_profit=0.15,
            stop_loss=0.05,
            max_holding_days=10,
        )

        # 添加分析器
        cerebro.addanalyzer(ValueRecorder, _name='valrec')
        # 运行后生成详细报告
        results = cerebro.run()
        # 提取数据
        strat = results[0]
        cerebro.plot()

        # ===== 自定义绘图 =====
        dates = pd.to_datetime([bt.num2date(x) for x in strat.data.datetime.array])
        values = strat.analyzers.valrec.values
        portfolio_values[f"{mode}"] = pd.Series(values, index=dates)


    df = pd.DataFrame(portfolio_values)
    df = df.sort_index()
    df.to_csv("strat_log.csv")

    plt.figure(figsize=(15, 7))

    for col in df.columns:
        plt.plot(df.index, df[col], label=col)

    plt.title('Portfolio Value Development')
    plt.xlabel('Date')
    plt.ylabel('Total Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()





