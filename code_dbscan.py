import numpy as np
import pandas as pd
df = pd.read_csv("data_clustering.csv")

labels = np.zeros(df.shape[0], dtype=float) - 1

print(labels)