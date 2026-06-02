# Train and save final Random Forest
# Van Hoi (Hendrick) Dang

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("acnc_classification.csv")

# split explanatory variables (X) from the response variable (y)
X = df.drop(columns=["Charity_Size", "target"])
y = df["target"]

# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# fit the final Random Forest on the training set
rf = RandomForestClassifier(n_estimators=1000, class_weight="balanced", random_state=0)
rf.fit(X_train.values, y_train.values)

# save the trained model and the feature order
joblib.dump({"model": rf, "features": X.columns.tolist()}, "rf_charity_size.joblib")
print("Saved trained model as rf_charity_size.joblib")