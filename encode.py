# Encode target variable and select features
# Van Hoi (Hendrick) Dang

import pandas as pd

df = pd.read_csv("acnc_final.csv", low_memory=False)

# encode target: Small=0, Medium=1, Large=2
size_map = {"Small": 0, "Medium": 1, "Large": 2}
df["target"] = df["Charity_Size"].map(size_map)

print(df["target"].value_counts().sort_index())

df.to_csv("acnc_encoded.csv", index=False)
print("Saved to acnc_encoded.csv")