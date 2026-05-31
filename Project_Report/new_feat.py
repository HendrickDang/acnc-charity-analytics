# Feature engineering
# Van Hoi (Hendrick) Dang

import pandas as pd

df = pd.read_csv("acnc_encoded.csv", low_memory=False)

# fill missing financial values with 0
financial_cols = [
    "Total_Revenue", "Total_Expenses", "Total_Assets",
    "Net_Assets", "Staff_FullTime", "Staff_PartTime", "Staff_Volunteers"
]
for col in financial_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# new features for classification
df["Revenue_per_Staff"] = df["Total_Revenue"] / (df["Staff_FullTime"] + df["Staff_PartTime"] + 1)
df["Total_Staff"]       = df["Staff_FullTime"] + df["Staff_PartTime"] + df["Staff_Volunteers"]
df["Has_AIS"]           = (df["Total_Revenue"] > 0).astype(int)
df["Rev_Exp_Ratio"]     = df["Total_Revenue"] / (df["Total_Expenses"] + 1)

print("New features created:")
print("  Revenue_per_Staff, Total_Staff, Has_AIS, Rev_Exp_Ratio")
print("\nMissing values after filling:")
print(df[financial_cols].isnull().sum().sum())

df.to_csv("acnc_features.csv", index=False)
print("Saved as acnc_features.csv")