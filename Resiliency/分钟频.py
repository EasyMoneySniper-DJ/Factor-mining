#!/usr/bin/python
"""
弹性因子计算，基于分钟频数据
"""
import os

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from statsmodels.tsa.filters.hp_filter import hpfilter
from datetime import datetime


## 读取本地数据
def load_data(parent_path, folder_name, list_folder_name, time):
    folder_path = os.path.join(parent_path, folder_name)
    list_folder_path = os.path.join(parent_path, list_folder_name)
    # 获取文件列表
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    # 定位索引
    # 拆分时间
    time_obj = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
    target_time = time_obj.strftime("%Y-%m-%d")
    target_file = f"{target_time}.csv"
    target_index = files.index(target_file)
    #截取文件
    start_index = target_index - 20
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
        df = pd.read_csv(file_path, usecols=['code', 'timestamp', 'close'])
        # 转成datetime格式
        df["timestamp"] = pd.to_datetime(df["timestamp"])
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
def cal_factor(data, time, win_min=60):
    """
    传入一个DataFrame，进行弹性因子计算
    :param data: 全部股票数据，必须包含["code", "timestamp", "log_close"]
    :param date: 计算日期 YYYY-MM-DD
    :param win_min: 时间窗口（数据长度），默认60
    :return: 包含了计算结果的DataFrame
    """
    data = data.copy()
    stock = data["code"].iloc[0]
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    factor_values = []  # 装因子值

    current_time = pd.to_datetime(time) # 转换日期
    # 截取需要的数据
    data_ex = data[(data["timestamp"] <= current_time)]
    data_ex = data_ex.tail(win_min)
    # print(data_ex)
    cycle_data_ex = data_ex["cycle"].dropna().values


    # 因子值计算逻辑
    f_it, Z_it = cal_fft(cycle_data_ex)
    D_it = len(cycle_data_ex)
    half_D_it = round(D_it / 2)
    factor_value = 1 / half_D_it * np.sum(2 * Z_it[:half_D_it] * f_it[:half_D_it])

    current_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    factor_values.append((stock, current_time, factor_value))

    return pd.DataFrame(factor_values, columns=["code", "timestamp", "factor_value"])


## 批量计算因子并保存
def pipeline(parent_path, folder_name, list_folder_name, time, win_min=60):
    """
    这一段是处理整个因子计算工作的函数
    :param parent_path: 数据父目录
    :param folder_name: 数据具体文件夹名称
    :param time: 想要计算的时间 %Y-%m-%d %H:%M:%S
    :param win_min: 时间窗口长度（数据长度），默认60条
    :return: 直接输出包含code, timestamp, factor_value的以date命名的csv文件
    """

    ## Step1: load data
    print("Loading data...")
    data = load_data(parent_path, folder_name, list_folder_name, time)

    ## Step2: 遍历当日全部股票并计算因子
    print("Calculating factors...")
    grouped = data.groupby("code")
    # 存结果
    all_results = []
    # 按个股计算因子
    for stock_code, group in grouped:
        print(f"Processing stock: {stock_code}")
        group = group.sort_values(by="timestamp").reset_index(drop=True)
        data_ex = group["log_close"].values
        # 判断数据长度够不够
        if len(data_ex) < win_min:
            print(f"数据长度不足，跳过{stock_code}")
            continue
        else:
            stock_data_ex = hp_filter(group)

        # 计算因子
        result = cal_factor(stock_data_ex, time, win_min)
        # if result is not None:
        all_results.append(result)

    ## Step3: 整合结果并输出csv
    final_result = pd.concat(all_results, ignore_index=True)
    # 保存结果到目标文件夹
    output_dir = os.path.join(parent_path, "result", "resiliency", f"{win_min}")
    os.makedirs(output_dir, exist_ok=True)
    formatted_time = pd.to_datetime(time).strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(output_dir, f"{formatted_time}.csv")
    final_result.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


## 主函数
if __name__ == "__main__":
    """
    定义参数，运行程序的地方
    :param parent_path: 父目录
    :param folder_name: 存放数据的位置
    :param list_folder_name: 存放股票列表的位置
    :param time: 目标时间 %Y-%m-%d %H:%M:%S
    :param win_min: 时间长度
    """
    # 设置数据路径
    parent_path = "D:\Quant\SHNF_Intern"
    folder_name = "data\stock_bfq_1m_price"
    list_folder_name = "data\list_stocks"
    # 目标日期
    time = "2025-01-15 15:00:00"
    # 时间长度
    win_min = 3600
    # 运行计算
    pipeline(parent_path, folder_name, list_folder_name, time, win_min)
