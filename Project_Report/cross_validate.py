# Cross validation: Naive Bayes, Decision Tree, Random Forest
# Van Hoi (Hendrick) Dang

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from statistics import mean

df = pd.read_csv("acnc_classification.csv")

# split explanatory variables (X) from the response variable (y)
X = df.drop(columns=["Charity_Size", "target"]).values
y = df["target"].values

# scoring metrics
scoring = ['accuracy', 'recall_macro', 'precision_macro', 'f1_macro']

# specify classifiers
classifiers = [
    ('Naive Bayes',    make_pipeline(StandardScaler(), GaussianNB())),
    ('Decision Tree',  DecisionTreeClassifier(class_weight="balanced", random_state=0)),
    ('Random Forest',  RandomForestClassifier(n_estimators=1000, class_weight="balanced", random_state=0)),
]

# run 10-fold cross-validation for each classifier
for name, clf in classifiers:
    scores = cross_validate(clf, X, y, cv=10, scoring=scoring)

    print("\nClassifier: %s" % name)
    print("Mean accuracy  : %.3f%%" % (mean(scores['test_accuracy'])*100))
    print("Mean precision : %.3f"   % (mean(scores['test_precision_macro'])))
    print("Mean recall    : %.3f"   % (mean(scores['test_recall_macro'])))
    print("Mean F1        : %.3f"   % (mean(scores['test_f1_macro'])))