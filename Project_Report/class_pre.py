# Classification Preprocessing
# Van Hoi (Hendrick) Dang

import pandas as pd

df = pd.read_csv("acnc_final.csv", low_memory=False)

print("Shape:", df.shape)

print("\nCharity_Size distribution:")
print(df["Charity_Size"].value_counts(dropna=False))

print("\nColumns available:")
print(df.columns.tolist())