import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pyentrp import entropy as ent

# --- Parameters ---
WINDOW_SIZE = 50
LAG = 1  # The time delay for SampEn, 1 is standard
EMBEDDING_DIM = 1  # 'm' parameter for SampEn
TOLERANCE_FACTOR = 0.2  # 'r' parameter: 0.2 * std(data) is common


class SampleEntropy:

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
                samp_en = ent.sample_entropy(window, EMBEDDING_DIM, LAG)
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
