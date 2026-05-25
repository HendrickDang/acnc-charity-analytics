# Model Evaluation Summary
# Van Hoi (Hendrick) Dang

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import pandas as pd

df = pd.read_csv("acnc_classification.csv")

# split explanatory variables (X) from the response variable (y)
X = df.drop(columns=["Charity_Size", "target"]).values
y = df["target"].values

# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# specify classifiers to compare
classifiers = [
    ("Naive Bayes",   GaussianNB(),                                                                       X_train_scaled, X_test_scaled),
    ("Decision Tree", DecisionTreeClassifier(class_weight="balanced", random_state=0),                    X_train,        X_test),
    ("Random Forest", RandomForestClassifier(n_estimators=1000, class_weight="balanced", random_state=0), X_train,        X_test),
]

# train each classifier and evaluate performance
for name, clf, Xtr, Xte in classifiers:
    clf.fit(Xtr, y_train)
    y_hat    = clf.predict(Xte)
    y_proba  = clf.predict_proba(Xte)

    acc  = metrics.accuracy_score(y_test, y_hat)*100
    prec = metrics.precision_score(y_test, y_hat, average="macro")
    rec  = metrics.recall_score(y_test, y_hat, average="macro")
    f1   = metrics.f1_score(y_test, y_hat, average="macro")
    auc  = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

    print("\nClassifier: ", name)
    print("\tAccuracy  : %.3f%%" % acc)
    print("\tPrecision : %.3f"   % prec)
    print("\tRecall    : %.3f"   % rec)
    print("\tF1        : %.3f"   % f1)
    print("\tROC-AUC   : %.3f"   % auc)
    print("\nClassification Report:")
    print(metrics.classification_report(y_test, y_hat, target_names=["Small", "Medium", "Large"]))