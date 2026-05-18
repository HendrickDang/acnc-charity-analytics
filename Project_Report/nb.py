# Naive Bayes Classifier
# Van Hoi (Hendrick) Dang

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

df = pd.read_csv("acnc_classification.csv")

# split explanatory variables (X) from the response variable (y)
X = df.drop(columns=["Charity_Size", "target"]).values
y = df["target"].values

# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# fit a GaussianNB classifier to the training set
clf = GaussianNB()
clf.fit(X_train, y_train)

# perform classification on the test set
y_hat = clf.predict(X_test)

# print performance
print("\nPrediction accuracy on the test dataset:")
print("{:.2%}".format(metrics.accuracy_score(y_test, y_hat)))
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_hat).sum()))

print("\nClassification Report:")
print(metrics.classification_report(y_test, y_hat, target_names=["Small", "Medium", "Large"]))

print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, y_hat))