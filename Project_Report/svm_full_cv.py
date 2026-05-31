# SVM
# Van Hoi (Hendrick) Dang

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from statistics import mean
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

# fit a SVM classifier to the training set
model = SVC(kernel="linear", class_weight="balanced", random_state=0)
model.fit(X_train_scaled, y_train)

# perform classification on the test set
y_hat = model.predict(X_test_scaled)

# print performance
print("Accuracy: %.3f%%" % (metrics.accuracy_score(y_test, y_hat)*100))
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_hat).sum()))

print("\nClassification Report:")
print(metrics.classification_report(y_test, y_hat, target_names=["Small", "Medium", "Large"]))

print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, y_hat))

# 5-fold cross-validation
scoring = ['accuracy', 'recall_macro', 'precision_macro', 'f1_macro']
pipe = make_pipeline(StandardScaler(), SVC(kernel="linear", class_weight="balanced", random_state=0))
scores = cross_validate(pipe, X, y, cv=5, scoring=scoring)

print("\n5-Fold Cross-Validation:")
print("Mean accuracy  : %.3f%%" % (mean(scores['test_accuracy'])*100))
print("Mean precision : %.3f"   % (mean(scores['test_precision_macro'])))
print("Mean recall    : %.3f"   % (mean(scores['test_recall_macro'])))
print("Mean F1        : %.3f"   % (mean(scores['test_f1_macro'])))