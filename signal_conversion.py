import numpy as np
import matplotlib
import pandas as pd
import pywt
from scipy.signal import lfilter

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


class SignalConversion:

    def __init__(self, file_path):
        self.file_path = file_path

    def FFTSmooth(self, low_pass_ratio=0.1):
        """
        对收盘价进行傅里叶低通滤波
        low_pass_ratio: 保留低频比例 (0~1)
        """
        # 1. 读取 CSV 数据
        df = pd.read_csv(self.file_path, parse_dates=['date'])
        if 'close' not in df.columns:
            raise ValueError("CSV 文件中必须包含 'close' 列")

        signal = df['close'].values
        n = len(signal)

        # 2. 傅里叶变换
        fft_values = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(n, d=1)

        # 3. 低通滤波：只保留前 low_pass_ratio 的低频
        cutoff = int(n * low_pass_ratio / 2)  # 保留左右对称部分
        fft_filtered = np.zeros_like(fft_values)
        fft_filtered[:cutoff] = fft_values[:cutoff]  # 低频部分保留
        fft_filtered[-cutoff:] = fft_values[-cutoff:]  # 对称低频部分保留

        # 4. 逆傅里叶变换恢复信号
        filtered_signal = np.fft.ifft(fft_filtered).real

        # 5. 绘图，用索引作为 x 轴
        plt.figure(figsize=(12, 6))
        plt.plot(range(n), signal, label='Original Signal')
        plt.plot(range(n), filtered_signal, label='Low-pass Filtered', color='red')
        plt.title("Low-pass Fourier Filter (X-axis as Index)")
        plt.xlabel("Sample Index")
        plt.ylabel("Close Price")
        plt.legend()
        plt.grid(True)
        plt.show()

        return filtered_signal

    def WaveletSmooth(self, wavelet='db4', level=None):
        """
        对收盘价进行小波去噪平滑
        wavelet: 小波类型，如 'db4', 'sym5', 'coif3'
        level: 分解层数，None 则自动选择最大层数
        """
        # 1. 读取 CSV 数据
        df = pd.read_csv(self.file_path)
        if 'close' not in df.columns:
            raise ValueError("CSV 文件中必须包含 'close' 列")

        signal = df['close'].values
        n = len(signal)

        # 2. 小波分解
        max_level = pywt.dwt_max_level(data_len=n, filter_len=pywt.Wavelet(wavelet).dec_len)
        if level is None or level > max_level:
            level = max_level

        coeffs = pywt.wavedec(signal, wavelet, level=level)

        # 3. 低频保留，高频系数置零（相当于低通滤波）
        coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]  # 保留 cA（低频），置 cD（高频）为 0

        # 4. 信号重构
        filtered_signal = pywt.waverec(coeffs, wavelet)
        filtered_signal = filtered_signal[:n]  # 重构可能比原信号长，截断

        # 5. 绘图，用索引作为 x 轴
        plt.figure(figsize=(12, 6))
        plt.plot(range(n), signal, label='Original Signal')
        plt.plot(range(n), filtered_signal, label='Wavelet Smoothed', color='red')
        plt.title(f"Wavelet Transform Smoothing ({wavelet}, level={level})")
        plt.xlabel("Sample Index")
        plt.ylabel("Close Price")
        plt.legend()
        plt.grid(True)
        plt.show()

        return filtered_signal

    def KalmanSmooth(self, process_variance=1e-5, measurement_variance=0.001):
        """
        对收盘价信号进行一维卡尔曼滤波平滑
        process_variance: 系统噪声方差 Q
        measurement_variance: 测量噪声方差 R
        """
        # 1. 读取 CSV 数据
        df = pd.read_csv(self.file_path)
        if 'close' not in df.columns:
            raise ValueError("CSV 文件中必须包含 'close' 列")

        signal = df['close'].values
        n = len(signal)

        # 2. 初始化卡尔曼滤波变量
        xhat = np.zeros(n)  # 滤波后的估计值
        P = np.zeros(n)  # 估计方差
        xhatminus = np.zeros(n)
        Pminus = np.zeros(n)
        K = np.zeros(n)  # 卡尔曼增益

        # 初始值
        xhat[0] = signal[0]
        P[0] = 1.0

        # 3. 卡尔曼滤波迭代
        for k in range(1, n):
            # 预测
            xhatminus[k] = xhat[k - 1]
            Pminus[k] = P[k - 1] + process_variance

            # 更新
            K[k] = Pminus[k] / (Pminus[k] + measurement_variance)
            xhat[k] = xhatminus[k] + K[k] * (signal[k] - xhatminus[k])
            P[k] = (1 - K[k]) * Pminus[k]

        # 4. 绘图
        plt.figure(figsize=(12, 6))
        plt.plot(range(n), signal, label='Original Signal')
        plt.plot(range(n), xhat, label='Kalman Filter Smoothed', color='red')
        plt.title(f"Kalman Filter Smoothing")
        plt.xlabel("Sample Index")
        plt.ylabel("Close Price")
        plt.legend()
        plt.grid(True)
        plt.show()

        return xhat

    def ZLowPassFilter(self, alpha=0.1):
        """
        使用一阶 IIR 数字滤波器实现 Z 域低通滤波
        alpha: 平滑系数 (0<alpha<1)，越小越平滑
        H(z) = (1-alpha)/(1 - alpha*z^-1)
        """
        # 1. 读取 CSV 数据
        df = pd.read_csv(self.file_path)
        if 'close' not in df.columns:
            raise ValueError("CSV 文件中必须包含 'close' 列")

        signal = df['close'].values
        n = len(signal)

        # 2. 构建数字滤波器系数
        b = [1 - alpha]  # 分子
        a = [1, -alpha]  # 分母

        # 3. 滤波
        filtered_signal = lfilter(b, a, signal)

        # 4. 绘图
        plt.figure(figsize=(12, 6))
        plt.plot(range(n), signal, label='Original Signal')
        plt.plot(range(n), filtered_signal, label=f'Z-Domain Low-pass (alpha={alpha})', color='red')
        plt.title("Z-Domain Low-pass Filtering")
        plt.xlabel("Sample Index")
        plt.ylabel("Close Price")
        plt.legend()
        plt.grid(True)
        plt.show()

        return filtered_signal


if __name__ == '__main__':
    signal_convert = SignalConversion('LC_20230721_20251030_Adjusted.csv')
    signal_convert.FFTSmooth()
