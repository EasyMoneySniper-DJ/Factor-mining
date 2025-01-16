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
def load_data(parent_path, folder_name, list_folder_name, date, win_days=60):
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
        df = pd.read_csv(file_path, usecols=['code', 'close'])
        # 从文件名提取日期并转换为日期格式
        df["date"] = pd.to_datetime(file.split(".csv")[0], format="%Y-%m-%d", errors="coerce")
        df = df[["date", "code", "close"]]
        # 按照valid_codes来过滤数据
        df = df[df["code"].isin(valid_codes)]
        if df.empty:
            continue
        data = pd.concat([data, df], ignore_index=True)

    # 确保列名无空格，类型统一
    data.columns = data.columns.str.strip()
    data["log_close"] = np.log(data["close"])
    print(data)
    return data

## 股票预处理

## HP滤波
def hp_filter(data):
    # data = data.sort_values(by="date").reset_index(drop=True)
    # data["log_close"] = np.log(data["close"])
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
def cal_factor(data, date, win_day=60):
    """
    传入一个DataFrame，进行弹性因子计算
    :param data: 全部股票数据，必须包含["code", "date", "close"]
    :param date: 计算日期 YYYY-MM-DD
    :param win_month: 时间窗口天数（数据长度），默认60个交易日
    :return: 包含了计算结果的DataFrame
    """
    data = data.copy()
    stock = data["code"].iloc[0]
    data["date"] = pd.to_datetime(data["date"])

    factor_values = []  # 装因子值

    current_date = pd.to_datetime(date) # 转换日期
    # 截取需要的数据
    data_ex = data[(data["date"] <= current_date)]
    data_ex = data_ex.tail(win_day)
    cycle_data_ex = data_ex["cycle"].dropna().values
    # # 判断数据长度
    # if len(cycle_data_ex) != win_day:
    #     print(f"数据长度不足，跳过{stock}")
    #     return None

    # 因子值计算逻辑
    f_it, Z_it = cal_fft(cycle_data_ex)
    D_it = len(cycle_data_ex)
    half_D_it = round(D_it / 2)
    factor_value = 1 / half_D_it * np.sum(2 * Z_it[:half_D_it] * f_it[:half_D_it])

    date = current_date.strftime("%Y-%m-%d")
    factor_values.append((stock, date, factor_value))

    return pd.DataFrame(factor_values, columns=["code", "date", "factor_value"])


## 批量计算因子并保存
def pipeline(parent_path, folder_name, list_folder_name, date, win_day=60):
    """
    这一段是处理整个因子计算工作的函数
    :param parent_path: 数据父目录
    :param folder_name: 数据具体文件夹名称
    :param date: 想要计算的日期 YYYY-MM-DD
    :param win_day: 时间窗口天数（数据长度），默认60个交易日
    :return: 直接输出包含code, date, factor_value的以date命名的csv文件
    """

    ## Step1: load data
    print("Loading data...")
    data = load_data(parent_path, folder_name, list_folder_name, date, win_day)

    ## Step2: 遍历当日全部股票并计算因子
    print("Calculating factors...")
    grouped = data.groupby("code")
    # 存结果
    all_results = []
    # 按个股计算因子
    for stock_code, group in grouped:
        print(f"Processing stock: {stock_code}")
        group = group.sort_values(by="date").reset_index(drop=True)
        data_ex = group["log_close"].values
        # 判断数据长度够不够
        if len(data_ex) < win_day:
            print(f"数据长度不足，跳过{stock_code}")
            continue
        else:
            stock_data_ex = hp_filter(group)

        # 计算因子
        result = cal_factor(stock_data_ex, date, win_day)
        # if result is not None:
        all_results.append(result)

    ## Step3: 整合结果并输出csv
    final_result = pd.concat(all_results, ignore_index=True)
    # 保存结果到目标文件夹
    output_dir = os.path.join(parent_path, "result", "resiliency", f"{win_day}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{date}.csv")
    final_result.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


## 主函数
if __name__ == "__main__":
    """
    定义参数，运行程序的地方
    :param parent_path: 父目录
    :param folder_name: 存放数据的位置
    :param list_folder_name: 存放股票列表的位置
    :param date: 目标日期
    :param win_day: 时间长度
    """
    # 设置数据路径
    parent_path = "D:\Quant\SHNF_Intern"
    folder_name = "data\stock_bfq_price"
    list_folder_name = "data\list_stocks"
    # 目标日期
    date = "2025-01-15"
    # 时间长度
    win_day = 120
    # 运行计算
    pipeline(parent_path, folder_name, list_folder_name, date, win_day)
