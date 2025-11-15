import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

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

    def get_threshold(self):
        pct = self.get_3min_pct()
        label = pct['pct_change_1'].dropna()  # 去掉 NaN

        # 右侧累积概率 0.35 对应左侧累积概率 0.65
        threshold = np.quantile(label, 0.6)

        print("右侧面积0.35对应的阈值:", threshold)
        # 0.0002714

        pass

    def mark_lable(self):
        df_3min = pd.read_csv("LC_20230731_20251030_3min.csv", parse_dates=['date'])
        df_3min['pct_change_1'] = df_3min['close'].pct_change(periods=1).shift(-1)
        threshold = 0.0002714
        df_3min['label'] = (df_3min['pct_change_1'] > threshold).astype(int)
        return df_3min

    def train_model_RF(self):
        df_3min = self.mark_lable()

        # ====== 选择特征列 ======
        # 根据你的数据结构自行修改，如果你没有其它特征，可先用 price 类特征测试
        feature_cols = [
            'open', 'high', 'low', 'close',
            'volume',
        ]
        feature_cols = [c for c in feature_cols if c in df_3min.columns]

        if not feature_cols:
            raise ValueError("❌ 没有找到可用特征列，请检查 CSV 中是否包含 open/high/low/close/vol/amount")

        X = df_3min[feature_cols]
        y = df_3min['label']

        # ====== 划分训练 / 测试 ======
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False  # 预测时间序列不打乱
        )

        # ====== 随机森林 ======
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )

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
    data_process = DataProcess("LC_20230731_20251030.csv")
    df = data_process.to_three_min()
    print(df.head(5))
