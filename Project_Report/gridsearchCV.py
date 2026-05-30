# GridSearch
# Van Hoi (Hendrick) Dang

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import pandas as pd

df = pd.read_csv("acnc_classification.csv")

# split explanatory variables (X) from the response variable (y)
X = df.drop(columns=["Charity_Size", "target"]).values
y = df["target"].values

# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# specify hyperparameters to search
param_grid = {
    "n_estimators": [100, 200, 500, 1000],
}

print("Finding the best hyperparameters for Random Forest ...")

model = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=0),
    param_grid=param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1
)

model.fit(X_train, y_train)

print("Best estimator parameters: ", model.best_params_)

# evaluate on test set
y_pred = model.predict(X_test)

print("--- Performance of Random Forest (best parameters) ---")
print("\tAccuracy: %.3f%%" % (metrics.accuracy_score(y_test, y_pred)*100))
print("\tF1 macro: %.3f"   % metrics.f1_score(y_test, y_pred, average="macro"))
print("\nClassification Report:")
print(metrics.classification_report(y_test, y_pred, target_names=["Small", "Medium", "Large"]))