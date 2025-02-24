#!/usr/bin/python
import day
import min
import pandas as pd
import os

def main():
    parent_path = ""
    list_stock_name = "data\list_stocks" #装股票列表的文件夹
    data_day_name = "data\stock_bfq_price" #装日频数据的
    data_1m_name = "data\stock_bfq_1m_price" #装分钟数据的

    data_path = os.path.join(parent_path, data_1m_name)
    data_list = sorted([f for f in os.listdir(data_path) if f.endswith(".csv")])
    data_list = data_list[21: -1]
    date_list = [filename.replace(".csv", "") for filename in data_list]
    print(date_list)

    # date = "2025-01-15"
    for date in date_list:
        print(f"Processing data: {date}>>>>>")
        df1 = day.pipeline_D(parent_path, data_day_name, list_stock_name, date)
        print(f"Processing 1min_data: {date}>>>>>")
        df2 = min.pipeline_M(parent_path, data_1m_name, list_stock_name, date)

        merged_df = pd.merge(df1, df2, on=["code", "date"], how="inner")

        output_dir = os.path.join(parent_path, "result", "resiliency")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{date}.csv")
        merged_df.to_csv(output_path, index=False)
        print("done")

if __name__ == "__main__":
    main()