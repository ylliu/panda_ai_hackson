import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pyentrp import entropy as ent
from scipy.spatial import cKDTree
from scipy.special import digamma

# --- Parameters ---
WINDOW_SIZE = 4
LAG = 1  # The time delay for SampEn, 1 is standard
EMBEDDING_DIM = 2  # 'm' parameter for SampEn
TOLERANCE_FACTOR = 0.2  # 'r' parameter: 0.2 * std(data) is common

data = np.array([0.1, 0.3, 0.2, -0.1])


class SampleEntropy:

    def sample_entropy_pyentrp(self, values, m=2, r_ratio=0.2):
        """
        使用 pyentrp 计算 Sample Entropy
        r_ratio: r = r_ratio * std
        """
        values = np.array(values)
        return ent.sample_entropy(values, m, r_ratio)

    import numpy as np

    def calculate_gram_mat(self, x, sigma):
        """calculate gram matrix for variables x
            Args:
            x: random variable with two dimensional (N,d).
            sigma: kernel size of x (Gaussian kernel)
        Returns:
            Gram matrix (N,N)
        """
        x = x.reshape(x.shape[0], -1)
        instances_norm = np.sum(x ** 2, axis=-1).reshape((-1, 1))
        dist = -2 * np.dot(x, x.T) + instances_norm + instances_norm.T
        return np.exp(-dist / sigma)

    def renyi_entropy(self, x, sigma, alpha):
        """
            Args:
            x: random variable with two dimensional (N,d).
            sigma: kernel size of x (Gaussian kernel)
            alpha: alpha value of renyi entropy
        Returns:
            renyi alpha entropy of x.
        """
        k = self.calculate_gram_mat(x, sigma)
        k = k / np.trace(k)

        # 计算对称矩阵的特征值
        eigv = np.abs(np.linalg.eigvalsh(k))
        eig_pow = eigv ** alpha
        entropy = (1 / (1 - alpha)) * np.log2(np.sum(eig_pow))
        return entropy

    def ksg_entropy_1d(self, x, k=5):
        x = np.asarray(x).reshape(-1, 1)
        n = len(x)

        tree = cKDTree(x)
        distances, _ = tree.query(x, k=k + 1)
        eps = distances[:, -1]

        nx = np.array([
            len(tree.query_ball_point(point, eps[i], p=np.inf)) - 1
            for i, point in enumerate(x)
        ])

        return digamma(n) - digamma(k) + np.mean(digamma(nx + 1)) + np.log(2)

    # 示例

    def sample_entropy(self, df):
        results = []
        # Calculate the overall standard deviation to set the tolerance 'r'
        overall_std = df['Value'].std()
        tolerance = TOLERANCE_FACTOR * overall_std

        for i in range(len(df) - WINDOW_SIZE + 1):
            # Define the window
            window = df['Value'].iloc[i: i + WINDOW_SIZE].values

            # --- X-axis Calculation: Sample Entropy ---
            try:
                # samp_en = nolds.sampen(window,
                #                        emb_dim=EMBEDDING_DIM,
                #                        tolerance=tolerance,
                #                        tau=LAG)
                samp_en = ent.sample_entropy(window, EMBEDDING_DIM, tolerance)
            except Exception:
                # Handle cases where SampEn cannot be computed (e.g., constant window)
                samp_en = np.nan

            # --- Y-axis Calculation: Representative Value (Mean) ---
            mean_value = np.mean(window)

            # Store the result
            results.append({'Entropy': samp_en[0], 'Mean_Value': mean_value})

        # Convert the results to a DataFrame for plotting
        entropy_df = pd.DataFrame(results).dropna()
        return samp_en[0]
