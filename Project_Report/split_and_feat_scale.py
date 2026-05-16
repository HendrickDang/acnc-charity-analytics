# Train/test split and feature scaling
# Van Hoi (Hendrick) Dang

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("acnc_final.csv", low_memory=False)

# encode target
size_map = {"Small": 0, "Medium": 1, "Large": 2}
df["target"] = df["Charity_Size"].map(size_map)

# fill missing financial values
financial_cols = [
    "Total_Revenue", "Total_Expenses", "Total_Assets",
    "Net_Assets", "Staff_FullTime", "Staff_PartTime", "Staff_Volunteers"
]
df[financial_cols] = df[financial_cols].fillna(0)
for col in financial_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# new features
df["Revenue_per_Staff"] = df["Total_Revenue"] / (df["Staff_FullTime"] + df["Staff_PartTime"] + 1)
df["Total_Staff"]       = df["Staff_FullTime"] + df["Staff_PartTime"] + df["Staff_Volunteers"]
df["Has_AIS"]           = (df["Total_Revenue"] > 0).astype(int)
df["Rev_Exp_Ratio"]     = df["Total_Revenue"] / (df["Total_Expenses"] + 1)

# define all features
state_ops = [
    "Operates_in_ACT", "Operates_in_NSW", "Operates_in_NT",
    "Operates_in_QLD", "Operates_in_SA", "Operates_in_TAS",
    "Operates_in_VIC", "Operates_in_WA"
]
purpose_cols = [
    "Preventing_or_relieving_suffering_of_animals", "Advancing_Culture",
    "Advancing_Education", "Advancing_Health",
    "Promote_or_oppose_a_change_to_law__government_poll_or_prac",
    "Advancing_natual_environment", "Promoting_or_protecting_human_rights",
    "Purposes_beneficial_to_ther_general_public_and_other_analogous",
    "Promoting_reconciliation__mutual_respect_and_tolerance",
    "Advancing_Religion", "Advancing_social_or_public_welfare",
    "Advancing_security_or_safety_of_Australia_or_Australian_public"
]
beneficiary_cols = [
    "Aboriginal_or_TSI", "Adults", "Aged_Persons", "Children",
    "Communities_Overseas", "Early_Childhood", "Ethnic_Groups", "Families",
    "Females", "Financially_Disadvantaged", "General_Community_in_Australia",
    "Males", "Migrants_Refugees_or_Asylum_Seekers", "Other_Beneficiaries",
    "Other_Charities", "People_at_risk_of_homelessness",
    "People_with_Chronic_Illness", "People_with_Disabilities",
    "Pre_Post_Release_Offenders", "Rural_Regional_Remote_Communities",
    "Unemployed_Person", "Veterans_or_their_families",
    "Victims_of_crime", "Victims_of_Disasters", "Youth"
]
engineered_cols = [
    "Num_States_Operated", "Num_Purposes", "Num_Beneficiaries",
    "Is_NT", "PBI", "HPC"
]
new_features = [
    "Revenue_per_Staff", "Total_Staff", "Has_AIS", "Rev_Exp_Ratio"
]

feature_cols = state_ops + purpose_cols + beneficiary_cols + engineered_cols + financial_cols + new_features

X = df[feature_cols].copy()
y = df["target"].copy()

# 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scale features — required for Naive Bayes (Gaussian) and SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Training set : {X_train.shape[0]:,} rows")
print(f"Test set     : {X_test.shape[0]:,} rows")
print(f"Features     : {X_train.shape[1]}")
print(f"\nClass distribution in training set:")
print(y_train.value_counts().sort_index().to_string())
print(f"\nClass distribution in test set:")
print(y_test.value_counts().sort_index().to_string())