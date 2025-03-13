#!/usr/bin/python
import backtrader as bt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class ValueRecorder(bt.Analyzer):
    def start(self):
        self.values = []

    def next(self):
        self.values.append(self.strategy.broker.getvalue())

class PairTrading(bt.Strategy):
    params = (
        ("window_size", 1000),
        ("threshold", 0.45),
        ("max_holding_days", 10),
        ("take_profit", 0.15),
        ("stop_loss", 0.05),
        ("trade_size", 100),
        ("trade_mode", "LONG"),
        ("upper_threshold", 0.003),
        ("lower_threshold", -0.003)
    )

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt}: {txt}")

    def __init__(self):
        # 创建不同数据源的交易信号记录器
        bt.observers.BuySell(self.datas[0])
        bt.observers.BuySell(self.datas[1])

        self.lookback = self.p.window_size
        self.trade_log = []

        self.data_main = self.datas[0]
        self.data_hedge = self.datas[1]

        self.entry_price_main = None
        self.entry_price_hedge = None
        self.entry_date_main = None
        self.entry_date_hedge = None      # 记录买入价和买入日期

        self.bar_num = 0
        self.df = self.datas[0].p.dataname
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df = self.df.set_index("date")
        # print(self.df.head())

        self.total_trades = 0
        self.daily_values = []  # 记录每日净值
        self.win_trades = 0
        self.pnl_list = []
        self.current_date = None

        self.entry_bar_main = None
        self.entry_bar_hedge = None

        self.last_trade_date = None  # 记录上一次交易日期

        self.daily_values = []  # 记录每日净值
        self.daily_dates = []  # 记录每日日期

        self.trade_start_date = "2016-01-01"

        self.excluded_cols = ['open', 'high', 'low', 'close', 'volume',
                              'future_return',
                              'excess_return', 'close_hedge']
        self.feat_cols = [col for col in self.df.columns if col not in self.excluded_cols]

    def notify_trade(self, trade):
        """记录交易盈亏"""
        if not trade.isclosed:
            return
        if trade.isclosed:
            pnl = trade.pnlcomm
            self.pnl_list.append(pnl)
            self.total_trades += 1
            self.win_trades += (pnl>0)

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

    def _train_model(self):
        feat_cols = ["realized_vol", "index_cap", "money_ratio", "return_volatility_rank", "volume_volatility", "vol_ma20", "vol_ma180", "mtm", "resiliency"]
        base_idx  = self.df.index[self.df.index.get_loc("2018-01-02")]
        current_index = self.df.index[self.df.index.get_loc(self.current_date)]
        # 确保 window_size 之前的数据存在
        start_date = self.df.index[self.df.index.get_loc(self.current_date) - self.p.window_size]
        end_date = self.df.index[self.df.index.get_loc(self.current_date) - 6]
        train_df = self.df.loc[:end_date]
        # print(train_df)
        if train_df.shape[0] == 0 or len(self.feat_cols) == 0:
            return
        X_train = train_df[self.feat_cols].values

        upper_threshold = self.p.upper_threshold
        lower_threshold = self.p.lower_threshold

        def label_func(x):
            if x > upper_threshold:
                return 1
            elif x < lower_threshold:
                return -1
            else:
                return 0

        y_train = train_df["excess_return"].apply(label_func)

        # 定义超参数搜索范围
        param_grid = {
            'logisticregression__C': [0.01, 0.1, 1, 10],
            'logisticregression__penalty': ['l1', 'l2']
        }

        # 训练模型
        pipeline = make_pipeline(StandardScaler(),
                                LogisticRegression(solver='saga', multi_class="ovr",
                                                   max_iter=1000, class_weight="balanced"))
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # 预测信号
        current_features = self.df.loc[[current_index], self.feat_cols].values
        prob = best_model.predict(current_features)[0]
        signal = prob



        print(signal)
        return signal

    def _cal_position(self, price_hedge, price_main, lot_size=1, cash_buffer=0.05):
        """
        优化后的仓位计算逻辑
        """
        total_value = self.broker.getvalue()
        # 最大仓位控制：不超过总资产的50%
        max_allocation = total_value * 0.5
        # 实际可用资金：根据现金缓冲调整
        available_cash = self.broker.get_cash() * (1 - cash_buffer)

        # 根据交易模式分配不同的目标资金
        if self.p.trade_mode in ["LONG_SHORT", "SHORT_BIG", "SHORT_CLOSE"]:
            # 这几种模式同时操作两个标的，均摊资金到两边
            allocation_per_leg = min(max_allocation * 0.5, available_cash * 0.5)
        else:
            # 如纯多头模式，仅操作一边，则全仓位使用
            allocation_per_leg = min(max_allocation, available_cash)

        # 定义一个计算可买手数的函数
        def calc_lots(target_value, price):
            if price <= 0:
                return 0
            # 计算可以购买的数量（原始数量）
            raw_lots = target_value / price
            # 调整为最小交易单位的整数倍
            adjusted_lots = max(int(raw_lots // lot_size) * lot_size, lot_size)
            # 限制最大购买量，确保不会超过可用资金
            max_lots = int(available_cash / price // lot_size) * lot_size
            return max(lot_size, min(adjusted_lots, max_lots))

        # 根据交易模式计算各标的的仓位
        if self.p.trade_mode == "LONG":
            position_hedge = calc_lots(allocation_per_leg, price_hedge)
            position_main = calc_lots(allocation_per_leg, price_main)
        else:
            # 对于多空、空平等模式，每个标的按分配资金计算仓位
            position_hedge = calc_lots(allocation_per_leg, price_hedge)
            position_main = calc_lots(allocation_per_leg, price_main)

        return position_hedge, position_main

    def _execute_trading(self, signal):
        pos_hedge = self.getposition(self.data_hedge).size
        print("Hedge Position: ", pos_hedge)
        pos_main = self.getposition(self.data_main).size
        print("Main Position: ", pos_main)

        price_hedge = self.data_hedge.close[0]
        price_main = self.data_main.close[0]

        size_hedge, size_main = self._cal_position(price_hedge, price_main)

        # 当预测信号为0时，代表在阈值区间，执行平仓逻辑：关闭所有持仓
        if signal == 0:
            if pos_main != 0:
                self.close(self.data_main)
                self._log_trade("CLOSE", "SMALL", price_main, pos_main, "CLOSE MAIN")
            if pos_hedge != 0:
                self.close(self.data_hedge)
                self._log_trade("CLOSE", "BIG", price_hedge, pos_hedge, "CLOSE HEDGE")
            return

        if self.p.trade_mode == "SHORT_BIG":
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
                    self.buy(self.data_hedge, size=size_hedge, exectype=bt.Order.Market)
                    self._log_trade("BUY", "BIG", price_hedge, size_hedge, "LONG BIG")
                    self.entry_price_hedge, self.entry_bar_hedge = price_hedge, len(self.data_hedge)




    def _check_take_profit_stop_loss(self):
        """检查止盈止损条件"""
        trigger = False
        # 今日持仓
        pos_hedge = self.getposition(self.data_hedge).size
        pos_main = self.getposition(self.data_main).size
        # 今日价格
        price_hedge = self.data_hedge.close[0]
        price_main = self.data_main.close[0]

        if self.p.trade_mode == "SHORT_BIG":
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

    def next(self):
        # 记录当前日期
        self.current_date = pd.Timestamp(self.datas[0].datetime.date(0))
        print(self.current_date)
        # 无条件记录每日日期和净值（保证两者长度一致）
        self.daily_dates.append(self.current_date)
        self.daily_values.append(self.broker.getvalue())
        base_date = pd.Timestamp("2018-01-02")
        # 如果数据不足，则直接返回
        if self.current_date < base_date:
            return

        # 如果当前日期不在训练数据中，则也返回（注意：此处不再记录日期，因为已经记录了）
        if self.current_date not in self.df.index:
            return

        # 以下为其它逻辑，比如止盈止损检查、交易信号等
        if self._check_take_profit_stop_loss():
            return

        if self._check_max_holding_days():
            return

        current_date = self.datetime.date(0)
        # 以上都没有，正常执行交易
        if self.last_trade_date and (current_date - self.last_trade_date).days < 5:
            return

        signal = self._train_model()
        if signal is not None:
            self._execute_trading(signal)
            self.last_trade_date = current_date

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

        # 用每日日期构造净值序列，保证索引对齐
        dates = pd.Series(self.daily_dates)
        values = pd.Series(self.daily_values, index=dates)

        # 计算收益率（自动丢弃第一个 NaN）
        returns = values.pct_change().dropna()

        # 将交易开始日期转换为 Timestamp（确保比较时类型一致）
        trade_start_date = pd.Timestamp(self.trade_start_date)

        # 过滤出实际交易期间的数据（注意：此时 values 与 returns 都使用相同的日期索引）
        trading_values = values.loc[values.index >= trade_start_date]
        trading_returns = returns.loc[returns.index >= trade_start_date]

        # 计算年化收益率（CAGR）
        total_years = len(trading_returns) / 252
        final_value = trading_values.iloc[-1]
        initial_value = trading_values.iloc[0]
        annualized_return = (final_value / initial_value) ** (1 / total_years) - 1

        # 计算夏普比率（避免 std 为零）
        sharpe_ratio = (trading_returns.mean() / trading_returns.std() * np.sqrt(252)
                        if trading_returns.std() > 0 else np.nan)

        # 计算最大回撤
        peak = trading_values.expanding().max()
        drawdown = (peak - trading_values) / peak
        max_drawdown = drawdown.max()

        # 计算 Calmar Ratio（年化收益 / 最大回撤）
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else np.nan

        # 计算 Sortino Ratio（只计算下行波动率）
        downside_returns = trading_returns[trading_returns < 0]
        downside_std = downside_returns.std()
        sortino_ratio = (trading_returns.mean() / downside_std * np.sqrt(252)
                         if downside_std > 0 else np.nan)

        # 计算盈亏比和胜率（假设 pnl_list 中只记录了实际交易的盈亏）
        wins = [x for x in self.pnl_list if x > 0]
        losses = [x for x in self.pnl_list if x < 0]
        profit_factor = abs(sum(wins) / sum(losses)) if losses else np.nan

        total_trades = len(self.pnl_list)
        winning_trades = len(wins)
        win_rate = winning_trades / total_trades if total_trades > 0 else np.nan

        # 计算最大连续亏损天数（使用实际交易期间的 returns）
        loss_streaks = (trading_returns < 0).astype(int).groupby((trading_returns >= 0).cumsum()).cumsum()
        max_consecutive_losses = loss_streaks.max()

        # 计算平均盈利和亏损
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        # 计算年度收益率（只计算实际交易期间）
        annual_returns = (1 + trading_returns).resample("Y").prod() - 1
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

        print("\n========== 分年度收益 ==========")
        for year, ret in annual_returns.items():
            print(f"{year}: {ret:.2%}")


class ExtendData(bt.feeds.PandasData):
    lines = (
        "realized_vol", "return_volatility_rank", "resiliency", "inflow_weipan", "vvix",
        "bond_return", "billboard_ratio", "money_ratio", "vol_ma20", "vol_ma180",
        "volume_drift", "future_return", "amp", "asset_bond_return_gap", "excess_return",
        "close_hedge", "mtm", "index_cap", "lj_corr", "amp_vol_divergence", "volume_volatility",
    )
    params = (
        ("dtformat", "%Y-%m-%d"),
        ("fromdate", datetime.datetime(2016, 5, 1)),
        ("todate", datetime.datetime(2025,2,1)),
        ("datetime", 0),
        ("open", 1),
        ("high", 2),
        ("low", 3),
        ("close", 4),
        ("volume", 5),
        ("openinterest", -1),
        ("realized_vol", -1), ("return_volatility_rank", -1), ("resiliency", -1), ("inflow_weipan", -1),
        ("vvix", -1), ("bond_return", -1), ("billboard_ratio", -1), ("money_ratio", -1),
        ("vol_ma20", -1), ("vol_ma180", -1), ("volume_drift", -1), ("future_return", -1), ("index_cap", -1),
        ("amp", -1), ("asset_bond_return_gap", -1), ("excess_return", -1), ("close_hedge", -1), ("mtm", -1),
        ("lj_corr", -1), ("amp_vol_divergence", -1), ("volume_volatility", -1),

    )

if __name__ == '__main__':
    cerebro = bt.Cerebro()

    # 添加策略，并传入参数
    cerebro.addstrategy(PairTrading,
                        # window_size=480,
                        # threshold=0.5,
                        # max_holding_days=10,
                        # transaction_cost=0.001
                        )

    # 读取 CSV 数据，假设其中包含所需的各列数据（包括 close_hedge 与额外特征）
    df_main = pd.read_csv("preprocess_weipan.csv", parse_dates=['date'])

    # 使用 PandasData作为主资产数据流（data0）
    data0 = ExtendData(
        dataname=df_main,
        name="MAIN",
        timeframe=bt.TimeFrame.Days,
        realized_vol=6,
        return_volatility_rank=7,
        resiliency=8,
        inflow_weipan=9,
        vvix=10,
        bond_return=11,
        billboard_ratio=12,
        money_ratio=13,
        index_cap=14,
        vol_ma20=15,
        vol_ma180=16,
        volume_drift=17,
        future_return=18,
        amp=19,
        asset_bond_return_gap=20,
        lj_corr=21,
        amp_vol_divergence=22,
        volume_volatility=23,
        excess_return=24,
        close_hedge=25,
        mtm=26,
        compression=1,
    )

    # 对冲资产数据流：这里复制 df，将 "close" 替换为 "close_hedge"
    df_hedge = pd.read_csv(r"hs300_price.csv", parse_dates=['date'])

    data1 = ExtendData(
        dataname=df_hedge,
        name="HEDGE",
        timeframe=bt.TimeFrame.Days,
        compression=1,
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
        cerebro.adddata(data0)
        cerebro.adddata(data1)

        cerebro.broker.set_cash(1000000)
        cerebro.broker.setcommission(commission=0.0003)  # 设置交易手续费
        print(">>>>>>>>>><<<<<<<<<<")
        print(f"===== {mode} 回测开始 =====")
        cerebro.addstrategy(
            PairTrading,
            trade_mode=mode,
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

