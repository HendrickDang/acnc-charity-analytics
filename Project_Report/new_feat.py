# Handle missing values and engineer new features
# Van Hoi (Hendrick) Dang

import pandas as pd
import numpy as np

df = pd.read_csv("acnc_final.csv", low_memory=False)

# encode target
size_map = {"Small": 0, "Medium": 1, "Large": 2}
df["target"] = df["Charity_Size"].map(size_map)

# fill missing financial values with 0
# justification: missing = no AIS return, most likely small/basic religious
# charities with minimal financial activity — 0 is appropriate
financial_cols = [
    "Total_Revenue", "Total_Expenses", "Total_Assets",
    "Net_Assets", "Staff_FullTime", "Staff_PartTime", "Staff_Volunteers"
]
df[financial_cols] = df[financial_cols].fillna(0)

# convert financial columns to numeric
for col in financial_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# new features for classification (fresh — not in A2)
# Revenue per staff member — proxy for organisational efficiency
df["Revenue_per_Staff"] = df["Total_Revenue"] / (df["Staff_FullTime"] + df["Staff_PartTime"] + 1)

# Total workforce size
df["Total_Staff"] = df["Staff_FullTime"] + df["Staff_PartTime"] + df["Staff_Volunteers"]

# Has financial data flag — 1 if AIS return exists, 0 if not
df["Has_AIS"] = (df["Total_Revenue"] > 0).astype(int)

# Revenue to expenses ratio — financial health indicator
df["Rev_Exp_Ratio"] = df["Total_Revenue"] / (df["Total_Expenses"] + 1)

print("New features created:")
print(f"  Revenue_per_Staff  : mean = {df['Revenue_per_Staff'].mean():.2f}")
print(f"  Total_Staff        : mean = {df['Total_Staff'].mean():.2f}")
print(f"  Has_AIS            : {df['Has_AIS'].value_counts().to_dict()}")
print(f"  Rev_Exp_Ratio      : mean = {df['Rev_Exp_Ratio'].mean():.2f}")

# verify no missing values remain
print(f"\nMissing values after handling: {df[financial_cols].isnull().sum().sum()}")