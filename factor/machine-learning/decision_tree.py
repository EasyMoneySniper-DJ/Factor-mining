#!/usr/bin/python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import factors

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

# ===== 读文件 =====
data_small = r"D:\Quant\SHNF_Intern\data\zz1000_price.csv"
data_big = r"D:\Quant\SHNF_Intern\data\hs300_price.csv"
target_col = ["date", "open", "high", "low", "close", "volume"]
df_small = pd.read_csv(data_small, usecols=target_col, parse_dates=["date"])
df_big = pd.read_csv(data_big, usecols=target_col, parse_dates=["date"])
# factors.cal_money_pct("000300.SH")
inflow_300 = pd.read_csv("inflow_300.csv", parse_dates=["date"])
inflow_1000 = pd.read_csv("inflow_1000.csv", parse_dates=["date"])
inflow_300["date"] = pd.to_datetime(inflow_300["date"])
inflow_1000["date"] = pd.to_datetime(inflow_1000["date"])

# ===== 交易信号 =====
for df in [df_small, df_big]:
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df["return"] = df["close"].pct_change()
    df["next_return"] = df["close"].pct_change(-5)

df_small = df_small.loc[df_big.index]  # 确保索引一致
df_small["signal"] = np.select(
    [
        (df_small["next_return"] >= 0) & (df_big["next_return"] <= 0),
        (df_small["next_return"] <= 0) & (df_big["next_return"] >= 0),
    ],
    [2, 0],
    default=1,
)


# ===== 构造因子 =====
for df in [df_small, df_big]:
    df["amp"] = factors.cal_amp(df)
    df["vol_ma20"] = factors.cal_ma(df["volume"], 20)
    df["vol_ma120"] = factors.cal_ma(df["volume"], 120)
    df["return_ma20"] = factors.cal_ma(df["return"], 20)
    df["return_ma120"] = factors.cal_ma(df["return"], 120)
    df["price_ma20"] = factors.cal_ma(df["close"], 20)
    df["price_ma120"] = factors.cal_ma(df["close"], 120)
    df["vol_drift"] = factors.cal_drift(df["volume"], df["vol_ma20"])
    df["std_20"] = factors.cal_std(df["return"], 20)
    df["std_120"] = factors.cal_std(df["return"], 120)
    df["sum_low"], df["sum_high"] = factors.cal_sum(df)
    df = factors.calc_rank_factor(df, 40)
    df.dropna(inplace=True)

    for factor in ["amp", "vol_ma20", "vol_ma120", "return_ma20", "return_ma120", "price_ma20",
                   "price_ma120", "vol_drift", "std_20", "std_120", "sum_low", "sum_high"]:
        df[f"{factor}_rank"] = factors.cal_rank(df[factor])

vvix_df = pd.read_csv("vvix.csv", parse_dates=["date"])
bond_df = pd.read_csv("bond.csv", parse_dates=["date"])
df = df_small.merge(vvix_df, on="date", how="left").merge(bond_df, on="date", how="left").merge(inflow_300, on="date", how="left")
df["reverse"] = factors.cal_reverse(df)
df = pd.merge(df, df_big, on="date", how="left", suffixes=("_small", "_big"))

df.dropna(inplace=True)
# ===== 开始学习 =====
print(df["signal"].value_counts(normalize=True))

trash_col = ["date", "signal", "next_return_small", "next_return_big"]
feat_col = ["amp_small", "std_20_small", "std_120_big", "amp_rank_small", "std_120_rank_big", "vvix",
                   "amp_big", "std_20_big", "std_120_rank_small"]

import statsmodels.api as sm
df["return_trend"], df["return_noise"] = sm.tsa.filters.hpfilter(df["next_return_small"], lamb=1600)

# 计算 return_noise 的标准差
noise_std = df["return_noise"].std()

# 设定阈值，例如 2 倍标准差
threshold = 2 * noise_std

# 过滤掉噪声过大的样本
df_filtered = df[df["return_noise"].abs() < threshold]

# 训练 LGBM 只使用低噪声数据
y = df_filtered["return_trend"]
X = df_filtered[feat_col]

# 生成数据（假设 X 是 10 个特征，y 是目标变量）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练决策树
tree_model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42)
tree_model.fit(X_train, y_train)

# 预测
y_pred = tree_model.predict(X_test)

# 在训练集上预测
y_train_pred = tree_model.predict(X_train)

# 计算并输出训练集的均方误差
train_mse = mean_squared_error(y_train, y_train_pred)
print(f"训练集均方误差: {train_mse}")

# 计算误差
mse = mean_squared_error(y_test, y_pred)
print(f"测试集均方误差: {mse}")

import joblib

# 将训练好的模型保存到文件
joblib.dump(tree_model, 'decision_tree_model.pkl')

print("模型已保存！")
