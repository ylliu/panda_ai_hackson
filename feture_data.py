import panda_data
import pandas as pd
from tqdm import tqdm
import tushare as ts


class FutureData:

    def test_panda_ai(self):
        symbol = 'LC2407'
        trade_date = '20240126'
        df = panda_data.get_market_min_data(
            symbol=symbol,
            start_date=trade_date,
            end_date=trade_date,
            symbol_type="future"
        )
        print(df.head(5))

    def download_main_contract_minute_data(self, mapping_df):

        """
        根据 fut_mapping 结果，自动逐日下载主力合约分钟数据，并整合成一个 DataFrame。
        """

        all_df_list = []

        for _, row in tqdm(mapping_df.iterrows(), total=len(mapping_df)):
            trade_date = row['trade_date']
            symbol = row['mapping_ts_code']  # 如 LC2407.GFE

            # PandaData 的 symbol 通常是 "LC2407"
            symbol_clean = symbol.replace(".GFE", "")

            try:
                df = panda_data.get_market_min_data(
                    symbol=symbol_clean,
                    start_date=trade_date,
                    end_date=trade_date,
                    symbol_type="future"
                )

                if df is not None and not df.empty:
                    df['trade_date'] = trade_date
                    df['symbol'] = symbol_clean
                    all_df_list.append(df)

            except Exception as e:
                print(f"[ERROR] {trade_date} - {symbol_clean}: {e}")

        # 合并
        if len(all_df_list) == 0:
            return pd.DataFrame()

        final_df = pd.concat(all_df_list, ignore_index=True)

        # 排序（按日期 + 时间）
        if 'datetime' in final_df.columns:
            final_df.sort_values(['trade_date', 'datetime'], inplace=True)

        return final_df

    def adjust_back_adjustment(self, df):
        """
        对分钟数据做“主力连续后复权”处理。
        df 格式必须包含：
            trade_date, symbol, datetime, close, open, high, low
        """

        if df.empty:
            return df

        # 保证按日期 + 时间排序
        df = df.sort_values(["trade_date", "date"]).copy()

        # 按天分组
        grouped = dict(tuple(df.groupby("trade_date")))

        # 用于记录累计复权因子（乘法形式）
        cumulative_factor = 1.0

        prev_last_close = None  # 前一天主力最后一分钟价格

        # 存储调整后的结果
        adjusted_list = []

        # 按日期遍历
        for trade_date, day_df in grouped.items():
            day_df = day_df.copy()

            # 获取当天第一分钟的 close
            first_close = day_df["close"].iloc[0]

            if prev_last_close is not None:
                # 计算切换因子（若连续则因子接近 1）
                factor = prev_last_close / first_close

                # 更新累计复权因子
                cumulative_factor *= factor

            # 记录今天复权前最后一分钟（用于下一天）
            prev_last_close = day_df["close"].iloc[-1]

            # 应用复权：后复权 = price * cumulative_factor
            for col in ["open", "high", "low", "close"]:
                day_df[col] = day_df[col] * cumulative_factor

            adjusted_list.append(day_df)

        # 合并
        final_df = pd.concat(adjusted_list, ignore_index=True)
        final_df.to_csv("LC_20230721_20251030_Adjusted.csv", index=False)
        return final_df

    def get_master_future(self):
        pro = ts.pro_api()
        future_list = pro.fut_mapping(
            ts_code='LC.GFE',
            start_date='20230101',
            end_date='20251030'
        )
        future_list.to_csv("fut_mapping_LC.csv", index=False)
        return future_list
