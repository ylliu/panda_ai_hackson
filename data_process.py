import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use('TkAgg')


class DataProcess:
    def __init__(self, file_path):
        self.file_path = file_path

    def to_three_min(self):
        import pandas as pd

        df = pd.read_csv(self.file_path, parse_dates=['date'])
        df.set_index('date', inplace=True)

        # 重采样为3分钟数据
        df_3min = df.resample('3T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
        df_3min.to_csv("LC_20230731_20251030_3min.csv", index=False)
        return df_3min

    def get_3min_pct(self):
        df_3min = pd.read_csv("LC_20230731_20251030_3min.csv", parse_dates=['date'])
        # 当前行与下一行的变化
        df_3min['pct_change_1'] = df_3min['close'].pct_change(periods=1).shift(-1)

        # 当前行与后两行的变化
        df_3min['pct_change_2'] = (df_3min['close'].shift(-2) - df_3min['close']) / df_3min['close']
        df_3min['pct_change_5'] = (df_3min['close'].shift(-5) - df_3min['close']) / df_3min['close']
        return df_3min

    def plot_data_distribution(self, label):
        pct = self.get_3min_pct()
        plt.figure(figsize=(10, 5))
        plt.hist(
            pct[label],
            bins=100,
            edgecolor='black',
            density=True  # <== y轴归一化到概率密度
        )
        plt.title('3-Minute Percentage Change Distribution')
        plt.xlabel('Percentage Change')
        plt.ylabel('Density (0-1)')
        # plt.show()
        plt.savefig(f'3min_pct_distribution{label}.png')

    def plot_data(self):
        df = pd.read_csv(self.file_path, parse_dates=['date'])
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['volume'], label='3-Minute Volume')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.title('3-Minute Resampled Volume Data')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    data_process = DataProcess("LC_20230731_20251030.csv")
    df = data_process.to_three_min()
    print(df.head(5))
