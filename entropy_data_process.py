import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
from tqdm import tqdm

from entropy import SampleEntropy

matplotlib.use('TkAgg')


class EntropyDataProcess:
    def __init__(self, file_path):
        self.file_path = file_path
        self.se = SampleEntropy()

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

    def calc_entropy_avg_as_threshold(self):
        df = pd.read_csv(self.file_path, parse_dates=['date'])
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        # 计算收益率
        df['return'] = df['close'].pct_change().shift(-1)
        df = df.dropna().reset_index(drop=True)

        # 每4分钟分组计算熵
        window_size = 4
        entropy_list = []
        for i in tqdm(range(0, len(df), window_size), desc="Calculating Entropy"):
            window = df['return'].iloc[i:i + window_size].values
            if len(window) < window_size:
                continue
            H = SampleEntropy().renyi_entropy(window, sigma=2, alpha=0.01)
            entropy_list.append(H)

        # 计算平均熵
        avg_entropy = np.nanmean(entropy_list)
        return avg_entropy

    def resample_to_entropy(self):
        """按熵阈值生成等熵 bar"""
        threshold = self.calc_entropy_avg_as_threshold()

        df = pd.read_csv(self.file_path, parse_dates=['date'])
        df['return'] = df['close'].pct_change().shift(-1)
        # 对close 去量纲
        df = df.dropna().reset_index(drop=True)

        bars = []
        temp_window = []
        for i, r in enumerate(df['return']):
            temp_window.append(r)
            window_array = np.array(temp_window)
            H = self.se.renyi_entropy(window_array, sigma=2, alpha=0.01)
            if H >= threshold:
                print(i)
                # 达到阈值生成一根 bar
                bar = {
                    'start_index': i - len(temp_window) + 1,
                    'end_index': i,
                    'open': df['close'].iloc[i - len(temp_window) + 1],
                    'close': df['close'].iloc[i],
                    'high': df['close'].iloc[i - len(temp_window) + 1:i + 1].max(),
                    'low': df['close'].iloc[i - len(temp_window) + 1:i + 1].min(),
                    'volume': df['volume'].iloc[i - len(temp_window) + 1:i + 1].sum(),
                    'entropy': H,
                    'count': len(temp_window)
                }
                bars.append(bar)
                temp_window = []  # 清空窗口，重新累积

        bars_df = pd.DataFrame(bars)
        bars_df.to_csv("LC_20230721_20251030_entropybar.csv", index=False)
        return bars_df

    def get_pct(self):
        df_volume = pd.read_csv("LC_20230721_20251030_entropybar.csv")
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
        df_volume = pd.read_csv("LC_20230721_20251030_entropybar.csv")
        df_volume['pct_change_1'] = df_volume['close'].pct_change(periods=1).shift(-1)
        threshold = self.get_threshold()
        df_volume['label'] = (df_volume['pct_change_1'] > threshold).astype(int)
        return df_volume

    def train_model_RF(self):
        df_3min = self.mark_label()

        # ====== 选择特征列 ======
        # 根据你的数据结构自行修改，如果你没有其它特征，可先用 price 类特征测试
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume'
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
        # 用Z score 去量纲

        X_train['close_zscore'] = (X_train['close'] - X_train['close'].mean()) / X_train['close'].std()
        X_train['open_zscore'] = (X_train['open'] - X_train['open'].mean()) / X_train['open'].std()
        X_train['high_zscore'] = (X_train['high'] - X_train['high'].mean()) / X_train['high'].std()
        X_train['low_zscore'] = (X_train['low'] - X_train['low'].mean()) / X_train['low'].std()
        X_train['volume_zscore'] = (X_train['volume'] - X_train['volume'].mean()) / X_train['volume'].std()
        X_test['close_zscore'] = (X_test['close'] - X_train['close'].mean()) / X_train['close'].std()
        X_test['open_zscore'] = (X_test['open'] - X_train['open'].mean()) / X_train['open'].std()
        X_test['high_zscore'] = (X_test['high'] - X_train['high'].mean()) / X_train['high'].std()
        X_test['low_zscore'] = (X_test['low'] - X_train['low'].mean()) / X_train['low'].std()
        X_test['volume_zscore'] = (X_test['volume'] - X_train['volume'].mean()) / X_train['volume'].std()

        XTrain = X_train[['close_zscore', 'open_zscore', 'high_zscore', 'low_zscore', 'volume_zscore']]
        XTest = X_test[['close_zscore', 'open_zscore', 'high_zscore', 'low_zscore', 'volume_zscore']]

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

        model.fit(XTrain, y_train)

        ####
        y_train_pred = model.predict(XTrain)
        print("🎯 RandomForest Train Accuracy:", accuracy_score(y_train, y_train_pred))
        print(classification_report(y_train, y_train_pred))
        # ====== 预测 ======
        y_pred = model.predict(XTest)

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
    data_process = EntropyDataProcess("LC_20230731_20251030.csv")
    df = data_process.to_min_of()
    print(df.head(5))
