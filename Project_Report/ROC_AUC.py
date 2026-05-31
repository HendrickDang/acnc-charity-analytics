# ROC-AUC Curve
# Van Hoi (Hendrick) Dang

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("acnc_classification.csv")

# split explanatory variables (X) from the response variable (y)
X = df.drop(columns=["Charity_Size", "target"]).values
y = df["target"].values

# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# binarize y for multiclass ROC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

# scale features for Naive Bayes and SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

class_names = ["Small", "Medium", "Large"]
colors = ["blue", "orange", "green"]

classifiers = [
    ("Naive Bayes",   GaussianNB(),                                                                       X_train_scaled, X_test_scaled),
    ("Decision Tree", DecisionTreeClassifier(class_weight="balanced", random_state=0),                    X_train,        X_test),
    ("SVM",           SVC(kernel="linear", class_weight="balanced", probability=True, random_state=0),    X_train_scaled, X_test_scaled),
    ("Random Forest", RandomForestClassifier(n_estimators=1000, class_weight="balanced", random_state=0), X_train,        X_test),
]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, (name, clf, Xtr, Xte) in zip(axes, classifiers):
    clf.fit(Xtr, y_train)
    y_proba = clf.predict_proba(Xte)

    for i, (cls, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = metrics.roc_curve(y_test_bin[:, i], y_proba[:, i])
        auc = metrics.auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, label="%s (AUC = %.2f)" % (cls, auc))

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — %s" % name)
    ax.legend(loc="lower right")

plt.suptitle("ROC Curves — All Classifiers", fontsize=14)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=300)
plt.show()

print("Saved as roc_curves.png")