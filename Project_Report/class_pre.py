# Classification Preprocessing Pipeline
# Van Hoi (Hendrick) Dang

import pandas as pd
import numpy as np

df = pd.read_csv("acnc_final.csv", low_memory=False)

print(f"Shape : {df.shape[0]:,} rows | {df.shape[1]} cols")
print(f"\nCharity_Size distribution:")
print(df["Charity_Size"].value_counts(dropna=False).to_string())
print(f"\nColumns available:")
print(df.columns.tolist())