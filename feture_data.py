import panda_data
import pandas as pd
from tqdm import tqdm
import tushare as ts


class FutureData:

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

    def get_master_future(self):
        pro = ts.pro_api()
        future_list = pro.fut_mapping(
            ts_code='LC.GFE',
            start_date='20230101',
            end_date='20251030'
        )
        return future_list
