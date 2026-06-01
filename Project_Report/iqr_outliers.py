# Outlier Detection (IQR Method)
# Van Hoi (Hendrick) Dang

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd

df = pd.read_csv("acnc_classification.csv")

X = df.drop(columns=["Charity_Size", "target"])
y = df["target"]

# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

financial_cols = [
    "Total_Revenue", "Total_Expenses", "Total_Assets",
    "Net_Assets", "Staff_FullTime", "Staff_PartTime", "Staff_Volunteers"
]

train = X_train.copy()
train["target"] = y_train.values

# apply IQR outlier detection only to AIS-reporting rows
mask = pd.Series(True, index=train.index)
ais_rows = train["Has_AIS"] == 1

for col in financial_cols:
    Q1 = train.loc[ais_rows, col].quantile(0.25)
    Q3 = train.loc[ais_rows, col].quantile(0.75)
    IQR = Q3 - Q1
    outlier_in_col = ais_rows & ~((train[col] >= Q1 - 1.5*IQR) & (train[col] <= Q3 + 1.5*IQR))
    mask = mask & ~outlier_in_col

print("Outliers detected in training set:", (~mask).sum())
print("Training rows before:", len(train))
print("Training rows after :", mask.sum())

# train Random Forest with and without outliers
rf = RandomForestClassifier(n_estimators=1000, class_weight="balanced", random_state=0)
rf.fit(X_train.values, y_train.values)
acc_before = metrics.accuracy_score(y_test, rf.predict(X_test.values))

train_clean = train[mask]
X_train_clean = train_clean.drop(columns=["target"]).values
y_train_clean = train_clean["target"].values

rf2 = RandomForestClassifier(n_estimators=1000, class_weight="balanced", random_state=0)
rf2.fit(X_train_clean, y_train_clean)
acc_after = metrics.accuracy_score(y_test, rf2.predict(X_test.values))

print("\nAccuracy WITH outliers   : %.3f%%" % (acc_before*100))
print("Accuracy WITHOUT outliers: %.3f%%" % (acc_after*100))