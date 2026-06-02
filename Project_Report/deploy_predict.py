# Load RF joblib file and predict
# Van Hoi (Hendrick) Dang

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn import metrics

# load the saved model and feature order
bundle = joblib.load("rf_charity_size.joblib")
model = bundle["model"]
features = bundle["features"]

df = pd.read_csv("acnc_classification.csv")

X = df[features]
y = df["target"]

# recreate the same test set used in training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# make predictions with the loaded model
y_hat = model.predict(X_test.values)

# confirm the loaded model gives the same performance
print("Accuracy of loaded model: %.3f%%" % (metrics.accuracy_score(y_test, y_hat)*100))

# show predicted size labels for the test charities
label_map = {0: "Small", 1: "Medium", 2: "Large"}
predicted = [label_map[p] for p in y_hat]
print("\nPredicted size distribution:")
print(pd.Series(predicted).value_counts())