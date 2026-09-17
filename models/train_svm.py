import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X = pd.read_csv("data/features.csv").values
y = pd.read_csv("data/labels.csv")["label"].values

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

# Fit scaler on TRAIN only, apply to all splits
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

best_c, best_acc = None, 0
for C in [0.1, 1, 10]:
    clf = SVC(kernel="rbf", C=C)
    clf.fit(X_train, y_train)
    val_acc = accuracy_score(y_val, clf.predict(X_val))
    print(f"C={C}: val accuracy = {val_acc:.3f}")
    if val_acc > best_acc:
        best_acc, best_c = val_acc, C

print(f"\nBest C = {best_c}")

X_trainval = np.concatenate([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])

final_clf = SVC(kernel="rbf", C=best_c)
final_clf.fit(X_trainval, y_trainval)

y_pred = final_clf.predict(X_test)
print("\nFinal Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["bonafide", "spoof"]))
print(confusion_matrix(y_test, y_pred))