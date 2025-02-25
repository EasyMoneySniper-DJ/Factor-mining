#!/usr/bin/python
import numpy as np
import pandas as pd
import os
from scipy.stats import percentileofscore

# 均线
def cal_ma(series, num):
    return series.rolling(num).mean()

# 偏离均线程度
def cal_drift(series1, series2):
    return (series1 / series2) - 1

# 日内振幅
def cal_amp(df):
    return (df["high"] / df["low"]) - 1

# 50ETF vvix指标计算
def cal_vvix():
    data_folder = r"D:\Quant\SHNF_Intern\data\50ETF_options"
    # output_file = "vvix_daily.csv"

    def cal_vvix(gamma, iv):
        if len(gamma) == 0 or len(iv) == 0:
            return np.nan

        weights = gamma / np.sum(gamma)
        vvix = np.sqrt(np.sum(weights * iv ** 2)) * 100
        return vvix

    vvix_results = []

    for file in sorted(os.listdir(data_folder)):
        if file.endswith(".csv"):
            file_path = os.path.join(data_folder, file)

            df = pd.read_csv(file_path)

            if "Gamma" in df.columns and "隐含波动率" in df.columns:
                df["Gamma"] = pd.to_numeric(df["Gamma"], errors="coerce")
                df["隐含波动率"] = pd.to_numeric(df["隐含波动率"], errors="coerce")

                df = df.dropna(subset=["Gamma", "隐含波动率"])

                gamma = df["Gamma"].values
                iv = df["隐含波动率"].values

                vvix_value = cal_vvix(gamma, iv)

                date = file.split(".")[0]

                vvix_results.append([date, vvix_value])

    vvix_df = pd.DataFrame(vvix_results, columns=["date", "vvix"])
    vvix_df["date"] = pd.to_datetime(vvix_df["date"])

    return vvix_df

# 计算国债20日收益率
def cal_bond_return():
    data_path = r"D:\Quant\SHNF_Intern\data\中证国债-历史价格.xls"
    df = pd.read_excel(data_path)
    target_cols = ["交易日期", "收盘价"]
    df = df[target_cols]
    df.columns = ["date", "close"]

    df["bond_return"] = df["close"].pct_change(20)
    df.dropna(inplace=True)
    df["date"] = pd.to_datetime(df["date"])

    return df[["date", "bond_return"]]

# 计算百分比排名
def rolling_pct(win):
    if len(win) < 1:
        return np.nan
    return percentileofscore(win, win[-1]) / 100

def cal_rank(series, window=750):
    return series.rolling(
        window=window,
        min_periods=1
    ).apply(rolling_pct, raw=True)

# 计算波动率
def cal_std(series, num):
    return series.rolling(num).std()

# 振幅因子切割
def cal_sum(df):
    df = df[["amp", "return"]].to_numpy()
    rolling_data = np.lib.stride_tricks.sliding_window_view(df, 240, axis=0)
    sum_low = np.array([
        np.nansum(win[np.argsort(win[:, 0])[:120], 1])
        for win in rolling_data
    ])
    sum_high = np.array([
        np.nansum(win[np.argsort(win[:, 0])[-120:], 1])
        for win in rolling_data
    ])

    return np.concatenate([[np.nan] * 239, sum_low]), np.concatenate([[np.nan] * 239, sum_high])

# 计算避险情绪
def cal_reverse(df):
    return df["close"].pct_change(20) - df["bond_return"]

# 历史收益率
def cal_his_return(df, num):
    return df["close"].pct_change(num)

# 指数大单资金流动情况
def cal_money_pct():
    index_weights_path = r"D:\Quant\SHNF_Intern\data\index_weights"
    money_flow_path = r"D:\Quant\SHNF_Intern\data\money_flow"
    results = []
    for file in os.listdir(index_weights_path):
        if file.endswith(".csv"):
            date = file.split(".csv")[0]
            index_weights = os.path.join(index_weights_path, file)
            df_weights = pd.read_csv(index_weights, encoding="gbk")
            df_300 = df_weights[df_weights["index_code"] == "000300.SH"][["code", "weight"]]
            money_flow = os.path.join(money_flow_path, f"{date}.csv")

            if os.path.exists(money_flow):
                df_money_flow = pd.read_csv(money_flow)
                df_money_flow = df_money_flow[["code", "net_pct_l"]]
                df_merge = df_300.merge(df_money_flow, on="code", how="left")

                df_merge["weighted_inflow"] = df_merge["weight"] * df_merge["net_pct_l"]
                inflow_300 = df_merge["weighted_inflow"].sum()

                results.append({"date": date, "inflow_300": inflow_300})


    df_results = pd.DataFrame(results)
    df_results.to_csv("300_inflow.csv", index=False)