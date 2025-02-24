#!/usr/bin/python
"""
因子分层测试框架（完整版）
包含功能：
1. 基础分层测试
2. 多空组合分析
3. 指数对冲分析
4. 多/指数组合分析
"""

# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


####################
# 配置模块
####################
class Config:

    # 字体配置
    font_path = r'C:\Windows\Fonts\msyh.ttc'  # 微软雅黑
    plt_style = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei'],  # 字体优先级
        'axes.unicode_minus': False  # 解决负号显示问题
    }

    # 路径配置
    price_dir = r""
    factor_dir = r""
    index_dir = r""
    output_dir = "./factor_strat_results"

    # 分析参数
    quantiles = 10  # 分层数量
    mad_threshold = 3  # MAD去极值阈值
    min_std = 1e-8
    enable_visualization = True  # 是否生成可视化图表
    factors_to_test = [  # 指定要测试的因子列表
        "resiliency_20", "resiliency_60", "resiliency_120",
        "resiliency_1m", "resiliency_1m_mean", "resiliency_1m_std",
        "resiliency_1m_W", "resiliency_1m_norm"
    ]
    hedge_method = 'simple'  # 对冲方法：simple/beta
    index_settings = {  # 指数配置
        "000300.SH": "沪深300",
        "000905.SH": "中证500",
        "000852.SH": "中证1000"
    }


####################
# 数据预处理模块
####################
class DataProcessor:
    @staticmethod
    def load_and_merge_data(cfg):
        """加载并合并因子与收益率数据"""
        factor_files = sorted([f for f in os.listdir(cfg.factor_dir) if f.endswith('.csv')])
        price_files = sorted(os.listdir(cfg.price_dir))
        price_index_map = {name: idx for idx, name in enumerate(price_files)}

        all_data = []
        for factor_file in tqdm(factor_files, desc="Processing factor files"):
            # 加载因子数据
            factor_path = os.path.join(cfg.factor_dir, factor_file)
            factor_df = pd.read_csv(factor_path)
            factor_date = factor_file.split(".csv")[0]

            # 获取对应价格数据
            price_today = f"{factor_date}.csv"
            try:
                target_idx = price_index_map[price_today]
                if target_idx + 1 >= len(price_files):
                    continue  # 跳过最后一个交易日
                next_price = price_files[target_idx + 1]
            except KeyError:
                continue

            # 合并收益率
            merge_price = DataProcessor._merge_price_data(
                cfg.price_dir, price_today, next_price
            )
            if merge_price is None:
                continue

            # 合并因子与收益
            merged = factor_df.merge(merge_price[["code", "return"]], on="code", how="inner")
            merged["date"] = pd.to_datetime(factor_date)
            all_data.append(merged)

        return pd.concat(all_data, ignore_index=True)

    @staticmethod
    def _merge_price_data(price_dir, today_file, next_file):
        """合并当日与下期价格数据"""
        try:
            today_df = pd.read_csv(os.path.join(price_dir, today_file))
            next_df = pd.read_csv(os.path.join(price_dir, next_file))
        except FileNotFoundError:
            return None

        merged = today_df.merge(
            next_df, on="code",
            suffixes=("_today", "_next"),
            how="inner"
        ).dropna(subset=["close_today", "close_next"])

        if len(merged) == 0:
            return None

        merged["return"] = merged["close_next"] / merged["close_today"] - 1
        return merged[["code", "return"]]

    @staticmethod
    def process_features(df, cfg):
        """特征工程处理（去极值+标准化）"""

        def mad_standardize(group):
            group = group.copy()
            for col in factor_cols:
                # MAD去极值
                median = group[col].median()
                mad = np.median(np.abs(group[col] - median))
                clip_min = median - cfg.mad_threshold * mad
                clip_max = median + cfg.mad_threshold * mad
                group[col] = np.clip(group[col], clip_min, clip_max)

                # 标准化
                mean, std = group[col].mean(), group[col].std()
                if std > cfg.min_std:
                    group[col] = (group[col] - mean) / std
                else:
                    group[col] = 0.0
            return group

        factor_cols = [c for c in df.columns if c not in ["code", "date", "return"]]
        if cfg.factors_to_test != 'all':
            factor_cols = [f for f in factor_cols if f in cfg.factors_to_test]

        return df.groupby("date", group_keys=False).apply(mad_standardize)


####################
# 指数处理模块
####################
class IndexProcessor:
    @staticmethod
    def load_index_data(cfg):
        """加载并处理指数收益率数据"""
        index_files = [f for f in os.listdir(cfg.index_dir) if f.endswith('.csv')]
        all_dfs = []

        for f in tqdm(index_files, desc="加载指数数据"):
            file_path = os.path.join(cfg.index_dir, f)
            try:
                file_date = pd.to_datetime(f.split(".csv")[0], format="%Y-%m-%d")
                df = pd.read_csv(file_path)
                df['date'] = file_date
                all_dfs.append(df)
            except Exception as e:
                print(f"加载文件{f}出错: {str(e)}")
                continue

        # 合并数据并计算收益率
        full_data = pd.concat(all_dfs, ignore_index=True)
        index_returns = []

        for code, group in tqdm(full_data.groupby('index_code'), desc="计算指数收益"):
            sorted_group = group.sort_values('date')
            sorted_group['return'] = sorted_group['close'].pct_change()
            pivot_df = sorted_group.pivot(index='date', columns='index_code', values='return')
            index_returns.append(pivot_df.dropna())

        return pd.concat(index_returns).sort_index().groupby(level=0).last()


####################
# 分析模块
####################
class FactorAnalyzer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.index_ret = IndexProcessor.load_index_data(cfg)
        os.makedirs(cfg.output_dir, exist_ok=True)

    def run_analysis(self, df):
        """执行多因子分层分析"""
        factor_cols = [c for c in df.columns if c in self.cfg.factors_to_test]

        for factor in tqdm(factor_cols, desc="Analyzing factors"):
            factor_df = df[["date", "code", factor, "return"]].dropna()
            self._single_factor_analysis(factor, factor_df)

    def _single_factor_analysis(self, factor_name, df):
        """单个因子分析流程"""
        # 基础分层
        df['stratum'] = pd.qcut(
            df[factor_name],
            q=self.cfg.quantiles,
            labels=False,
            duplicates='drop'
        )

        # 保存基础结果
        self._plot_basic_results(factor_name, df)

        # 多空对冲分析
        self._hedge_analysis(factor_name, df)

    def _hedge_analysis(self, factor_name, df):
        """执行对冲分析"""
        # 基础分层收益
        daily_ret = df.groupby(["date", "stratum"])["return"].mean().unstack()

        # 计算多/空
        relative_ret = (1+daily_ret[0]).cumprod() / (1+daily_ret[9]).cumprod() - 1

        # 多/沪深300
        hs300_dates = daily_ret.index.intersection(self.index_ret["000300.SH"].index)
        hs300_low_ret = daily_ret[0].loc[hs300_dates]
        hs300_ret = self.index_ret["000300.SH"].loc[hs300_dates]
        relative_hs300 = (1+hs300_low_ret).cumprod() / (1+hs300_ret).cumprod() - 1

        # 多/中证1000
        zz1000_dates = daily_ret.index.intersection(self.index_ret["000852.SH"].index)
        zz1000_low_ret = daily_ret[0].loc[zz1000_dates]
        zz1000_ret = self.index_ret["000852.SH"].loc[zz1000_dates]
        relative_zz1000 = (1+zz1000_low_ret).cumprod() / (1+zz1000_ret).cumprod() - 1

        # 多/中证500
        zz500_dates = daily_ret.index.intersection(self.index_ret["000905.SH"].index)
        zz500_low_ret = daily_ret[0].loc[zz500_dates]
        zz500_ret = self.index_ret["000905.SH"].loc[zz500_dates]
        relative_zz500 = (1+zz500_low_ret).cumprod() / (1+zz500_ret).cumprod() - 1

        # ===== 绘制相对收益曲线 =====
        plt.figure(figsize=(14,8))

        # 多/空
        relative_ret.plot(
            label="多/空",
            color="#4C6643",
            linewidth=2
        )

        # 多/沪深300
        relative_hs300.plot(
            label="多/沪深300",
            color="#AFCFA6",
            linewidth=2
        )

        # 多/中证500
        relative_zz500.plot(
            label="多/中证500",
            color="#F5D44B",
            linewidth=2
        )

        # 多/中证1000
        relative_zz1000.plot(
            label="多/中证1000",
            color="#D47828",
            linewidth=2
        )

        plt.title(f"{factor_name}-相对收益")
        plt.ylabel("相对累计收益")
        plt.axhline(0, color="gray", linestyle=":")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.cfg.output_dir, f"{factor_name}_相对收益.png"))
        plt.close()

    def _plot_basic_results(self, factor_name, df):
        """绘制基础结果图表"""
        # 因子分布箱线图
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='stratum', y=factor_name, data=df)
        plt.title(f'{factor_name} - 因子分布')
        plt.savefig(os.path.join(self.cfg.output_dir, f'{factor_name}_分布.png'))
        plt.close()

        # 分层累计复利曲线
        plt.figure(figsize=(14, 8))
        # 各层累计收益
        cum_ret = df.groupby(["date", "stratum"])["return"].mean().unstack()
        cum_ret = (1+cum_ret).cumprod()-1

        # 绘制分层曲线
        for stratum in sorted(cum_ret.columns):
            cum_ret[stratum].plot(
                label=f"{stratum+1}",
                alpha=0.7 if stratum not in [0,9] else 1.0,
                linewidth=2 if stratum in [0,9] else 1
            )
        plt.title(f"{factor_name}-分层测试")
        plt.ylabel("累计收益率")
        plt.legend(title="分层", ncol=3)
        plt.grid(True)
        plt.savefig(os.path.join(self.cfg.output_dir, f'{factor_name}_分层测试.png'))
        plt.close()

####################
# 主流程
####################
if __name__ == "__main__":
    # 初始化字体配置
    import matplotlib as mpl
    import matplotlib.font_manager as fm

    cfg = Config()
    plt.style.use('seaborn-v0_8')
    mpl.rcParams.update(cfg.plt_style)

    # 注册指定字体
    try:
        fontprop = fm.FontProperties(fname=cfg.font_path)
        fm.fontManager.addfont(cfg.font_path)
        mpl.rcParams['font.sans-serif'] = [fontprop.get_name()] + mpl.rcParams['font.sans-serif']
    except:
        print(f"⚠️ 自定义字体加载失败: {cfg.font_path}, 使用系统默认中文字体")

    # 数据预处理
    print("=> 数据预处理阶段")
    processor = DataProcessor()
    raw_df = processor.load_and_merge_data(cfg)
    processed_df = processor.process_features(raw_df, cfg)

    # 执行分析
    print("\n=> 分层分析阶段")
    analyzer = FactorAnalyzer(cfg)
    analyzer.run_analysis(processed_df)

    print(f"\n✅ 分析完成！结果已保存至 {cfg.output_dir}")
