class DataProcess:
    def __init__(self, file_path):
        self.file_path = file_path

    def to_three_min(self):
        import pandas as pd

        df = pd.read_csv(self.file_path, parse_dates=['date'])
        df.set_index('date', inplace=True)

        # 重采样为3分钟数据
        df_3min = df.resample('3T').agg({
            'volume': 'sum'
        }).dropna().reset_index()

        return df_3min


if __name__ == '__main__':
    data_process = DataProcess("LC_20230731_20251030.csv")
    df = data_process.to_three_min()
    print(df.head(5))
