import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc

matplotlib.use('TkAgg')


class VolumeDataProcess:
    def __init__(self, file_path):
        self.file_path = file_path

    def to_min_of(self, minute):
        import pandas as pd

        df = pd.read_csv(self.file_path, parse_dates=['date'])
        df.set_index('date', inplace=True)

        # 重采样为3分钟数据
        df_min = df.resample(f'{minute}T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
        df_min.to_csv(f"LC_20230721_20251030_{minute}min.csv", index=False)
        return df_min

    def resample_to_volume(self, volume_threshold):
        """
        根据给定成交量阈值生成等量 Bar
        volume_threshold: 每根Bar的目标成交量
        """
        # 读取数据
        df = pd.read_csv(self.file_path, parse_dates=['date'])

        # 初始化变量
        bars = []
        cum_vol = 0
        o, h, l, c = df['open'].iloc[0], df['high'].iloc[0], df['low'].iloc[0], df['close'].iloc[0]
        start_time = df['date'].iloc[0]

        for idx, row in df.iterrows():
            price_open, price_high, price_low, price_close, vol = \
                row['open'], row['high'], row['low'], row['close'], row['volume']

            # 累积成交量
            cum_vol += vol
            h = max(h, price_high)
            l = min(l, price_low)
            c = price_close

            # 达到阈值生成Bar
            if cum_vol >= volume_threshold:
                end_time = row['date']
                time_delta = (end_time - start_time).total_seconds()

                bars.append({
                    'open': o,
                    'high': h,
                    'low': l,
                    'close': c,
                    'volume': cum_vol,
                    'start_time': start_time,
                    'end_time': end_time,
                    'time_delta_sec': time_delta
                })

                # 重置累计变量
                cum_vol = 0
                o, h, l, c = price_open, price_high, price_low, price_close
                start_time = row['date']

        # 处理最后一根未满阈值的Bar
        if cum_vol > 0:
            end_time = df['date'].iloc[-1]
            time_delta = (end_time - start_time).total_seconds()

            bars.append({
                'open': o,
                'high': h,
                'low': l,
                'close': c,
                'volume': cum_vol,
                'start_time': start_time,
                'end_time': end_time,
                'time_delta_sec': time_delta
            })

        volume_bar_df = pd.DataFrame(bars)
        volume_bar_df.to_csv(f"LC_20230721_20251030_volumebar_{volume_threshold}.csv", index=False)

        return volume_bar_df

    def plot_volume_bars(self, volume_bar_df, title='Volume Bars'):
        """
        绘制等量 Bar 的K线图
        volume_bar_df: 包含 ['open','high','low','close','volume','start_time'] 的DataFrame
        """
        # 先转换时间为matplotlib可识别的浮点数
        ohlc = volume_bar_df.copy()
        ohlc['date_float'] = mdates.date2num(ohlc['start_time'])
        ohlc_data = ohlc[['date_float', 'open', 'high', 'low', 'close']].values

        # 创建图表
        plt.figure(figsize=(12, 6))
        ax = plt.gca()

        # 绘制K线
        candlestick_ohlc(ax, ohlc_data, width=0.0005, colorup='g', colordown='r')

        # 设置X轴时间格式
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('volume_bars.png')
        plt.show()

    def get_pct(self):
        df_volume = pd.read_csv("LC_20230721_20251030_volumebar_5800.csv")
        # 当前行与下一行的变化
        df_volume['pct_change_1'] = df_volume['close'].pct_change(periods=1).shift(-1)

        # 当前行与后两行的变化
        df_volume['pct_change_2'] = (df_volume['close'].shift(-2) - df_volume['close']) / df_volume['close']
        df_volume['pct_change_5'] = (df_volume['close'].shift(-5) - df_volume['close']) / df_volume['close']
        return df_volume

    def plot_volume_over_time(self, minute):
        df = self.to_min_of(minute)
        self.plot_data_distribution(minute, df, 'volume')

    def plot_data_distribution(self, minute, df, label):
        # 计算中位数
        print(f"Plotting distribution for {minute}-Minute {label}")
        V_bar = df[label].quantile(0.5)
        print(f"Median volume (50% quantile) as threshold: {V_bar}")

        # 计算平均值
        mean_value = df[label].mean()
        print(f"Mean volume: {mean_value}")

        plt.figure(figsize=(10, 5))
        plt.hist(
            df[label],
            bins=100,
            edgecolor='black',
            density=True  # <== y轴归一化到概率密度
        )
        plt.title(f'{minute}-Minute {label} Distribution')
        plt.xlabel(f'{label}')
        plt.ylabel('Density (0-1)')
        # 绘制中位数线
        plt.axvline(V_bar, color='red', linestyle='--', label=f'Median={V_bar:.2f}')
        plt.legend()
        # 绘制平均值线
        plt.axvline(mean_value, color='green', linestyle='--', label=f'Mean={mean_value:.2f}')
        plt.legend()
        # plt.show()
        plt.savefig(f'{minute}min_{label}_distribution.png')

    def plot_data(self):
        df = pd.read_csv(self.file_path, parse_dates=['date'])
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['volume'], label='3-Minute Volume')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.title('3-Minute Resampled Volume Data')
        plt.legend()
        plt.show()

    def get_threshold(self):
        pct = self.get_pct()
        label = pct['pct_change_1'].dropna()  # 去掉 NaN

        # 右侧累积概率 0.35 对应左侧累积概率 0.65
        threshold = np.quantile(label, 0.6)

        print("右侧面积0.35对应的阈值:", threshold)
        # 0.0002714
        return threshold

    def mark_label(self):
        df_volume = pd.read_csv("LC_20230721_20251030_volumebar_4780.csv")
        df_volume['pct_change_1'] = df_volume['close'].pct_change(periods=1).shift(-1)
        threshold = self.get_threshold()
        df_volume['label'] = (df_volume['pct_change_1'] > threshold).astype(int)
        return df_volume

    def train_model_RF(self):
        df_3min = self.mark_label()

        # ====== 选择特征列 ======
        # 根据你的数据结构自行修改，如果你没有其它特征，可先用 price 类特征测试
        feature_cols = [
            'open', 'high', 'low', 'close', 'time_delta_sec'
        ]
        feature_cols = [c for c in feature_cols if c in df_3min.columns]

        if not feature_cols:
            raise ValueError("❌ 没有找到可用特征列，请检查 CSV 中是否包含 open/high/low/close/volume")

        X = df_3min[feature_cols]
        y = df_3min['label']

        n_samples = len(X)
        train_size = int(n_samples * 0.7)  # 70% 训练集，30% 测试集

        # 手动划分
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]

        # # ====== 划分训练 / 测试 ======
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.3, shuffle=False  # 预测时间序列不打乱
        # )

        # ====== 随机森林 ======
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # 处理类别不平衡
        )

        # model = RandomForestClassifier(
        #     n_estimators=100,
        #     criterion='gini',
        #     random_state=42,
        #     class_weight='balanced'
        # )

        model.fit(X_train, y_train)

        ####
        y_train_pred = model.predict(X_train)
        print("🎯 RandomForest Train Accuracy:", accuracy_score(y_train, y_train_pred))
        print(classification_report(y_train, y_train_pred))
        # ====== 预测 ======
        y_pred = model.predict(X_test)

        # ====== 输出结果 ======
        print("🎯 RandomForest Accuracy:", accuracy_score(y_test, y_pred))
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))
        df_con = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })

        # ====== 特征重要性 ======
        fi = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\n🔍 Feature Importance:")
        print(fi)

        return model, fi


if __name__ == '__main__':
    data_process = VolumeDataProcess("LC_20230731_20251030.csv")
    df = data_process.to_min_of()
    print(df.head(5))
