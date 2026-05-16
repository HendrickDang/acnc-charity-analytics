# Final classification dataset
# Van Hoi (Hendrick) Dang

import pandas as pd

df = pd.read_csv("acnc_features.csv", low_memory=False)

# define feature columns
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
financial_cols = [
    "Total_Revenue", "Total_Expenses", "Total_Assets",
    "Net_Assets", "Staff_FullTime", "Staff_PartTime", "Staff_Volunteers"
]
new_features = [
    "Revenue_per_Staff", "Total_Staff", "Has_AIS", "Rev_Exp_Ratio"
]

feature_cols = state_ops + purpose_cols + beneficiary_cols + engineered_cols + financial_cols + new_features

# keep only features and target
df_out = df[feature_cols + ["Charity_Size", "target"]].copy()

df_out.to_csv("acnc_classification.csv", index=False)

print("Rows:", len(df_out))
print("Cols:", df_out.shape[1])
print("\nTarget distribution:")
print(df_out["target"].value_counts().sort_index())
print("Saved to acnc_classification.csv")