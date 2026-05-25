# Confusion Matrix Visualisation
# Van Hoi (Hendrick) Dang

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("acnc_classification.csv")

# split explanatory variables (X) from the response variable (y)
X = df.drop(columns=["Charity_Size", "target"]).values
y = df["target"].values

# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# scale features for Naive Bayes
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

class_names = ["Small", "Medium", "Large"]

# fit and evaluate Naive Bayes
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred_nb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()
plt.title("Confusion Matrix — Naive Bayes")
plt.savefig("cm_naive_bayes.png", dpi=300)
plt.show()

# fit and evaluate Decision Tree
dt = DecisionTreeClassifier(class_weight="balanced", random_state=0)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

cm = confusion_matrix(y_test, y_pred_dt)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()
plt.title("Confusion Matrix — Decision Tree")
plt.savefig("cm_decision_tree.png", dpi=300)
plt.show()

# fit and evaluate Random Forest
rf = RandomForestClassifier(n_estimators=1000, class_weight="balanced", random_state=0)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()
plt.title("Confusion Matrix — Random Forest")
plt.savefig("cm_random_forest.png", dpi=300)
plt.show()

print("Saved → cm_naive_bayes.png, cm_decision_tree.png, cm_random_forest.png")