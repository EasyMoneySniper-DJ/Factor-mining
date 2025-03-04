#!/usr/bin/python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def prepare_data(file_path):
    """
    读取并预处理数据：
    1) 生成 future_5d_return
    2) 生成 target
    3) 删除缺失值
    """
    df = pd.read_csv(file_path, parse_dates=['date']).sort_values('date').set_index('date')
    # 生成未来5日收益率
    df['future_5d_return'] = (df['close'].shift(-5) - df['close']) / df['close']
    # 方向标签：未来5日收益率大于0为1，否则为0
    df['target'] = (df['future_5d_return'] > 0).astype(int)

    # 去除缺失
    df.dropna(inplace=True)
    return df

def backtest(df, params):
    """
    回测逻辑（简化版）：
    - 仅在每隔 5 天 (t % 5 == 0) 才训练模型、生成交易信号并进行开/平仓操作
    - 每日仍更新组合净值：若持仓，则净值跟随收盘价变动；若空仓，则保持不变
    - 若持仓天数超过 max_holding_days，强制平仓
    """
    window_size = params['window_size']         # 用于训练模型的窗口长度
    threshold = params['threshold']             # 信号阈值
    max_holding_days = params['max_holding_days']
    cost = params['transaction_cost']

    # 初始化：信号、仓位、组合净值
    signals = pd.Series(np.nan, index=df.index)
    positions = pd.Series(0, index=df.index)
    portfolio = pd.Series(np.nan, index=df.index)

    # 状态记录
    in_trade = False
    entry_day = None

    # 初始资金
    initial_capital = 1.0

    # 设定要排除的列（价格、成交量、标签等）
    excluded_cols = ['open','high','low','close','volume','future_5d_return','target', 'future_return', 'excess_return']
    feat_col = [c for c in df.columns if c not in excluded_cols]

    # 为了有足够的数据训练模型，回测从第 window_size 天开始
    start_idx = window_size
    # 初始化组合净值
    portfolio.iloc[start_idx] = initial_capital

    # 逐日回测
    for t in tqdm(range(start_idx + 1, len(df))):
        # ========== 1) 更新组合净值 ==========
        if positions.iloc[t - 1] == 1:
            # 持仓 => 净值随价格变动
            portfolio.iloc[t] = portfolio.iloc[t - 1] * (df['close'].iloc[t] / df['close'].iloc[t - 1])
        else:
            # 空仓 => 净值保持不变
            portfolio.iloc[t] = portfolio.iloc[t - 1]

        # ========== 2) 检查是否超过最大持仓天数，若超则强制平仓 ==========
        if in_trade and entry_day is not None:
            holding_days = t - entry_day
            if holding_days >= max_holding_days:
                # 强制平仓，扣除平仓成本
                portfolio.iloc[t] *= (1 - cost)
                positions.iloc[t] = 0
                in_trade = False
                entry_day = None
                # 当天操作结束，跳过交易决策
                continue

        # ========== 3) 每隔 5 天才做一次模型训练 & 交易决策 ==========
        if t % 5 == 0:
            # 3.1) 训练数据区间： [t-window_size, t-1]（不含 t）
            train_start = t - window_size
            train_end = t-5  # 到 t（不含 t 本身）
            X_train = df[feat_col].iloc[train_start:train_end - 1].values
            y_train = df['target'].iloc[train_start:train_end - 1].values

            # 如果没有特征可用，跳过
            if X_train.shape[1] == 0:
                # 保持前一日仓位
                positions.iloc[t] = positions.iloc[t - 1]
                signals.iloc[t] = signals.iloc[t - 1]
                continue

            # 3.2) 训练模型
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(penalty='l1', C=0.1, solver='saga', max_iter=1000)
            )
            model.fit(X_train, y_train)

            # 3.3) 预测当日 (t) 的看涨概率
            current_features = df[feat_col].iloc[t].values.reshape(1, -1)
            prob = model.predict_proba(current_features)[0][1]
            signals.iloc[t] = prob

            # 3.4) 交易逻辑
            if (not in_trade) and (prob > threshold):
                # 开仓 => 扣除进场成本
                portfolio.iloc[t] *= (1 - cost)
                positions.iloc[t] = 1
                in_trade = True
                entry_day = t
            elif in_trade and (prob <= threshold):
                # 平仓 => 扣除平仓成本
                portfolio.iloc[t] *= (1 - cost)
                positions.iloc[t] = 0
                in_trade = False
                entry_day = None
            else:
                # 信号不触发交易，保持前一日仓位
                positions.iloc[t] = positions.iloc[t - 1]

        else:
            # 不到 5 日，不做训练和交易决策，保持前一日仓位和信号
            positions.iloc[t] = positions.iloc[t - 1]
            signals.iloc[t] = signals.iloc[t - 1]

    # ========== 4) 补全 NaN 并计算基准 ==========
    portfolio.fillna(method='ffill', inplace=True)
    positions.fillna(method='ffill', inplace=True)
    signals.fillna(method='ffill', inplace=True)

    benchmark = df['close'] / df['close'].iloc[start_idx]

    # 整理结果
    results = pd.DataFrame({
        'strategy': portfolio,
        'signals': signals,
        'positions': positions,
        'benchmark': benchmark
    })
    return results

def analyze_performance(results):
    # 计算最大回撤
    cumulative_max = results['strategy'].cummax()
    drawdown = (results['strategy'] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()

    # 年化夏普比率（简单估计252个交易日）
    returns = results['strategy'].pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()

    print(f"最终净值: {results['strategy'].iloc[-1]:.4f}")
    print(f"最大回撤: {max_drawdown * 100:.2f}%")
    print(f"年化夏普比率: {sharpe_ratio:.2f}")

    # 绘图
    plt.figure(figsize=(14, 7))
    plt.plot(results.index, results['strategy'], label='Strategy')
    plt.plot(results.index, results['benchmark'], label='Benchmark', alpha=0.7)

    # 标记进场时点：positions 从 0 -> 1
    buy_signals = results[(results['positions'] == 1) & (results['positions'].shift(1) == 0)].index
    plt.scatter(buy_signals, results.loc[buy_signals, 'strategy'], marker='^', color='red', label='Entry')

    plt.title("Strategy Performance (Trade every 5 days)")
    plt.legend()
    plt.show()


# ============== 参数 ==============
params = {
    'window_size': 480,        # 训练窗口长度
    'threshold': 0.5,          # 信号阈值
    'max_holding_days': 10,    # 最大持仓天数
    'transaction_cost': 0.001  # 单边交易成本
}

if __name__ == "__main__":
    df = prepare_data("preprocess.csv")
    results = backtest(df, params)
    analyze_performance(results)
