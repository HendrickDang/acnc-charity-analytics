# Random Forest Classifier
# Van Hoi (Hendrick) Dang

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv("acnc_classification.csv")

# split explanatory variables (X) from the response variable (y)
X = df.drop(columns=["Charity_Size", "target"]).values
y = df["target"].values

# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# fit a RandomForestClassifier to the training set
rf = RandomForestClassifier(n_estimators=1000, class_weight="balanced", random_state=0)
rf.fit(X_train, y_train)

# perform classification on the test set
y_pred = rf.predict(X_test)

# print performance
print("Accuracy: %.3f%%" % (metrics.accuracy_score(y_test, y_pred)*100))
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

print("\nClassification Report:")
print(metrics.classification_report(y_test, y_pred, target_names=["Small", "Medium", "Large"]))

print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred))