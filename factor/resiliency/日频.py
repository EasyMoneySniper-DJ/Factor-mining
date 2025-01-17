#!/usr/bin/python
"""
弹性因子计算：基于日频数据
"""

import os

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from statsmodels.tsa.filters.hp_filter import hpfilter


## 读取本地数据
def load_data(parent_path, folder_name, list_folder_name, date, win_days):
    folder_path = os.path.join(parent_path, folder_name)
    list_folder_path = os.path.join(parent_path, list_folder_name)
    # 获取文件列表
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    # 定位索引
    target_file = f"{date}.csv"
    target_index = files.index(target_file)
    #截取文件
    start_index = target_index - (win_days + 10)
    select_files = files[start_index: target_index+1]

    data = pd.DataFrame()

    # 加载list_stock目标日期文件code列
    code_file = os.path.join(list_folder_path, target_file)
    if not os.path.exists(code_file):
        raise FileExistsError(f"List stock file{code_file} not found")

    code_list = pd.read_csv(code_file, usecols=["code"], encoding="utf-8")
    valid_codes = set(code_list["code"].dropna().unique())

    # 顺序读取文件
    for file in select_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, usecols=['date', 'code', 'close'])
        # 从文件名提取日期并转换为日期格式
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "code", "close"]]
        # 按照valid_codes来过滤数据
        df = df[df["code"].isin(valid_codes)]
        if df.empty:
            continue
        data = pd.concat([data, df], ignore_index=True)

    # 确保列名无空格，类型统一
    data.columns = data.columns.str.strip()
    data["log_close"] = np.log(data["close"])
    # print(data)
    return data

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

    #提取股票代码
    code = data["code"].iloc[0]

    return pd.DataFrame({
        "code": [code],
        "factor_value": [factor_value]
    })


## 批量计算因子并保存
def pipeline(parent_path, folder_name, list_folder_name, date):
    # 定义数据长度
    win_days = [20, 60, 120]
    # 存结果
    all_results = pd.DataFrame()

    ## Step1: load data
    print("Loading data...")
    data = load_data(parent_path, folder_name, list_folder_name, date, max(win_days))

    ## Step2: 遍历当日全部股票并计算因子
    print("Calculating factors...")
    grouped = data.groupby("code")
    # 按个股计算因子
    for stock_code, group in grouped:
        print(f"Processing stock: {stock_code}")
        group = group.sort_values(by="date").reset_index(drop=True)
        #取max(win_days)长度的数据
        group = group.tail(max(win_days))
        #确保数据长度
        if len(group) < max(win_days):
            print(f"数据长度不足，跳过{stock_code}")
            continue
        #应用HP滤波
        stock_data_ex = hp_filter(group)
        #根据不同时间窗口计算因子
        stock_result = pd.DataFrame({
            "code": [stock_code],
            "date": [date]
        })
        for win_day in win_days:
            #再按照具体的win_day取数据
            data_ex = stock_data_ex.copy()
            data_ex = data_ex.tail(win_day)
            # print(len(data_ex))
            factor_df = cal_factor(data_ex)
            #判断是否为空（感觉可以去掉，前面的确保数据长度操作应该已经确保df非空）
            if not factor_df.empty:
                stock_result[f"resiliency_{win_day}"] = factor_df["factor_value"].values[0]
            else:
                stock_result[f"resiliency_{win_day}"] = np.nan
        #合并结果
        all_results = pd.concat([all_results, stock_result], ignore_index=True)

    ## Step3: 整合结果并输出csv
    # 保存结果到目标文件夹
    output_dir = os.path.join(parent_path, "result", "resiliency")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{date}.csv")
    all_results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


## 主函数
if __name__ == "__main__":
    # 设置数据路径
    parent_path = "D:\Quant\SHNF_Intern"
    folder_name = "data\stock_bfq_price"
    list_folder_name = "data\list_stocks"
    # 目标日期
    date = "2025-01-14"
    # 运行计算
    pipeline(parent_path, folder_name, list_folder_name, date)
