import numpy as np
import pandas as pd

df = pd.read_csv("data_clustering.csv")
df = df.to_numpy()
print(np.where(np.linalg.norm(df - df[5], axis=1) < 5)[0])