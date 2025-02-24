#!/usr/bin/python
"""弹性因子计算，基于分钟频数据
计算逻辑：当日因子值为20日因子值求平均
加入20天平均 20天标准差 截面上两个值标准化之后相加
"""
import os
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from statsmodels.tsa.filters.hp_filter import hpfilter

def load_data(parent_path, folder_name, list_folder_name, file_name):
    folder_path = os.path.join(parent_path, folder_name)
    list_folder_path = os.path.join(parent_path, list_folder_name)

    # 加载list_stock目标日期文件code列
    code_file = os.path.join(list_folder_path, file_name)
    if not os.path.exists(code_file):
        raise FileExistsError(f"List stock file{code_file} not found")

    valid_codes = pd.read_csv(code_file, usecols=["code"], encoding="utf-8")["code"].dropna().unique()

    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path, usecols=["code", "timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["code"].isin(valid_codes)]
    df["log_close"] = np.log(df["close"])
    return df

def hp_filter(data):
    data["trend"], data["cycle"] = hpfilter(data["log_close"], lamb=1600)
    return data

def cal_fft(sequence):
    D = len(sequence)
    fft_result = fft(sequence)
    frequencies = fftfreq(D)
    positive_freq = frequencies[:D // 2]
    positive_amp = abs(fft_result[:D // 2]) / D
    return positive_freq, positive_amp

def cal_factor(data):
    data_ex = data["cycle"].dropna().values
    f_it, Z_it = cal_fft(data_ex)
    D_it = len(data_ex)
    half_D_it = round(D_it / 2)
    factor_value = 1 / half_D_it * np.sum(2 * Z_it[:half_D_it] * f_it[:half_D_it])
    code = data["code"].iloc[0]
    # return pd.DataFrame({"code": [code], "resiliency_1m": [factor_value]})
    return factor_value

def cal_factor_W(data, t_date):
    result = pd.DataFrame()
    data = data.sort_values(by=["code", "date"])
    grouped = data.groupby("code")
    for stock_code, group in grouped:
        stock_result = pd.DataFrame({"code": [stock_code], "date": [t_date]})
        stock_result["resiliency_1m"] = group["resiliency_1m"].iloc[-1]
        stock_result["resiliency_1m_mean"] = group["resiliency_1m"].mean()
        stock_result["resiliency_1m_std"] = group["resiliency_1m"].std()
        group = group.sort_values(by="resiliency_1m", ascending=False)
        split_index = len(group) // 2
        high_sum = group.iloc[:split_index]["resiliency_1m"].sum()
        low_sum = group.iloc[split_index:]["resiliency_1m"].sum()
        stock_result["resiliency_1m_W"] = high_sum - low_sum
        result = pd.concat([result, stock_result], ignore_index=True)
    return result

def cal_factor_norm(data):
    data["resiliency_1m_mean_norm"] = (data["resiliency_1m_mean"] - data["resiliency_1m_mean"].mean()) / data["resiliency_1m_mean"].std()
    data["resiliency_1m_std_norm"] = (data["resiliency_1m_std"] - data["resiliency_1m_std"].mean()) / data["resiliency_1m_std"].std()
    data["resiliency_1m_norm"] = data["resiliency_1m_mean_norm"] + data["resiliency_1m_std_norm"]
    data.drop(columns=["resiliency_1m_mean_norm", "resiliency_1m_std_norm"], inplace=True)
    return data

def pipeline_M(parent_path, folder_name, list_folder_name, date):
    folder_path = os.path.join(parent_path, folder_name)
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    target_file = f"{date}.csv"
    target_index = files.index(target_file)
    start_index = max(0, target_index - 20)
    select_files = files[start_index: target_index + 1]
    # print(select_files)
    all_result = pd.DataFrame()
    for file in select_files:
        target_date = file.split(".")[0]
        data = load_data(parent_path, folder_name, list_folder_name, file)
        # print(data)
        grouped = data.groupby("code")
        result = pd.DataFrame()
        for stock_code, group in grouped:
            group = group.sort_values(by="timestamp").reset_index(drop=True)
            stock_data = hp_filter(group)
            factor_value = cal_factor(stock_data)
            stock_result = pd.DataFrame({"code": [stock_code], "date": [target_date]})
            stock_result["resiliency_1m"] = [factor_value]
            result = pd.concat([result, stock_result], ignore_index=True)
        all_result = pd.concat([all_result, result], ignore_index=True)

    all_result = cal_factor_W(all_result, date)
    all_result = cal_factor_norm(all_result)
    return all_result
