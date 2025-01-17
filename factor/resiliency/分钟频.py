#!/usr/bin/python
"""
弹性因子计算，基于分钟频数据
计算逻辑：当日因子值为20日因子值求平均
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from statsmodels.tsa.filters.hp_filter import hpfilter
from datetime import datetime


## 读取本地数据
def load_data(parent_path, folder_name, list_folder_name, file_name):
    folder_path = os.path.join(parent_path, folder_name)
    list_folder_path = os.path.join(parent_path, list_folder_name)

    data = pd.DataFrame()

    # 加载list_stock目标日期文件code列
    code_file = os.path.join(list_folder_path, file_name)
    if not os.path.exists(code_file):
        raise FileExistsError(f"List stock file{code_file} not found")

    code_list = pd.read_csv(code_file, usecols=["code"], encoding="utf-8")
    valid_codes = set(code_list["code"].dropna().unique())

    #读文件
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path, usecols=["code", "timestamp", "close"])
    #timestamp列转datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    #过滤数据
    df = df[df["code"].isin(valid_codes)]
    # 确保列名无空格，类型统一
    df.columns = df.columns.str.strip()
    df["log_close"] = np.log(df["close"])
    # print(data)
    return df

## 股票预处理

## HP滤波
def hp_filter(data):
    cycle, trend = hpfilter(data["log_close"], lamb=1600)
    data["trend"] = trend
    data["cycle"] = cycle
    return data


## 离散傅里叶变换
def cal_fft(sequence):
    D = len(sequence)
    fft_result = fft(sequence)  # 傅里叶变换
    frequencies = fftfreq(D)  # 频率
    positive_freq = frequencies[:D // 2]  # 取正频部分
    positive_amp = abs(fft_result[:D // 2]) / D  # 归一化幅度
    return positive_freq, positive_amp


## 计算弹性因子
def cal_factor(data):

    data = data.copy()
    data_ex = data["cycle"].dropna().values

    # 因子值计算逻辑
    f_it, Z_it = cal_fft(data_ex)
    D_it = len(data_ex)
    half_D_it = round(D_it / 2)
    factor_value = 1 / half_D_it * np.sum(2 * Z_it[:half_D_it] * f_it[:half_D_it])

    #股票代码
    code = data["code"].iloc[0]

    return pd.DataFrame({
        "code": [code],
        "resiliency_1m": [factor_value]
    })

##处理原始因子值
def cal_factor_mean(data, date, win_size=20):
    data = data.copy()
    data = data.sort_values(by=["code", "date"])
    data["resiliency_1m_mean"] = (
        data.groupby("code")["resiliency_1m"]
        .transform(lambda x: x.rolling(window=win_size, min_periods=win_size).mean())
    )
    # data = data.dropna()
    data_ex = data[data["date"] == date].dropna()
    return data_ex


## 批量计算因子并保存
def pipeline(parent_path,folder_name, list_folder_name, date):
    """
    由于我们想要得到的最后的因子值是由原始因子值求20天平均获得，因此应该先循环load data计算原始因子值，再rolling求平均。
    目前想到的办法是将整个步骤都放到for循环里面处理，先拿到一个装有40天原始因子值的DataFrame，再rolling。
    """
    all_result = pd.DataFrame()

    ## Step1: load data
    folder_path = os.path.join(parent_path, folder_name)
    list_folder_path = os.path.join(parent_path, list_folder_name)
    # 获取文件列表
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    # 定位索引
    target_file = f"{date}.csv"
    target_index = files.index(target_file)
    #截取文件
    start_index = target_index - 40
    select_files = files[start_index: target_index+1]

    print("Loading data...")
    for file in select_files:
        result = pd.DataFrame()
        target_date = file.split(".")[0]
        data = load_data(parent_path, folder_name, list_folder_name, file)

        ## Step2: 遍历当日全部股票并计算因子
        print("Calculating factors...")
        grouped = data.groupby("code")

        # 按个股计算因子
        for stock_code, group in grouped:
            print(f"Processing stock: {stock_code}")
            group = group.sort_values(by="timestamp").reset_index(drop=True)
            data_ex = group["log_close"].values
            stock_data_ex = hp_filter(group)
            stock_result = pd.DataFrame({
                "code": [stock_code],
                "date": [target_date]
            })

            # 计算因子
            factor_df = cal_factor(stock_data_ex)
            stock_result["resiliency_1m"] = factor_df["resiliency_1m"].values[0]

            result = pd.concat([result, stock_result], ignore_index=True)

        # Step3: 整合结果并输出csv
        all_result = pd.concat([all_result, result], ignore_index=True)

    all_result = cal_factor_mean(all_result, date)
    # 保存结果到目标文件夹
    output_dir = os.path.join(parent_path, "result", "resiliency")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{date}_1m.csv")
    all_result.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


## 主函数
if __name__ == "__main__":
    # 设置数据路径
    parent_path = "D:\Quant\SHNF_Intern"
    folder_name = "data\stock_bfq_1m_price"
    list_folder_name = "data\list_stocks"
    # 目标日期
    date = "2025-01-15"
    # 运行计算
    pipeline(parent_path, folder_name, list_folder_name, date)
