import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the training pool (seen attack types only)
X_train_full = pd.read_csv("data/features_gentrain.csv").values
y_train_full = pd.read_csv("data/labels_gentrain.csv")["label"].values

# Load the unseen-attack test pool
X_unseen = pd.read_csv("data/features_genunseen.csv").values
y_unseen = pd.read_csv("data/labels_genunseen.csv")["label"].values

# Split training pool into trainval/test (normal test set — SEEN attacks)
X_trainval, X_test_seen, y_trainval, y_test_seen = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# Scale — fit on trainval only
scaler = StandardScaler()
X_trainval = scaler.fit_transform(X_trainval)
X_test_seen = scaler.transform(X_test_seen)
X_unseen = scaler.transform(X_unseen)  # same scaler, never refit

# Train SVM with GridSearchCV
svm_grid = GridSearchCV(
    SVC(kernel="rbf", probability=True),
    param_grid={"C": [0.1, 1, 10]},
    cv=5,
    scoring="accuracy"
)
svm_grid.fit(X_trainval, y_trainval)
print("Best SVM params:", svm_grid.best_params_)
final_svm = svm_grid.best_estimator_

# Evaluate on SEEN-attack test set (normal generalization, within-distribution)
y_pred_seen = final_svm.predict(X_test_seen)
print("\n=== SVM: Test on SEEN attack types ===")
print("Accuracy:", accuracy_score(y_test_seen, y_pred_seen))
print(classification_report(y_test_seen, y_pred_seen, target_names=["bonafide", "spoof"]))
print(confusion_matrix(y_test_seen, y_pred_seen))

# Evaluate on UNSEEN-attack test set (the actual generalization test)
y_pred_unseen = final_svm.predict(X_unseen)
print("\n=== SVM: Test on UNSEEN attack types (A16, A09) ===")
print("Accuracy:", accuracy_score(y_unseen, y_pred_unseen))
print(classification_report(y_unseen, y_pred_unseen, target_names=["bonafide", "spoof"]))
print(confusion_matrix(y_unseen, y_pred_unseen))


from sklearn.ensemble import RandomForestClassifier

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20]},
    cv=5,
    scoring="accuracy"
)
rf_grid.fit(X_trainval, y_trainval)
print("Best RF params:", rf_grid.best_params_)
final_rf = rf_grid.best_estimator_

y_pred_seen_rf = final_rf.predict(X_test_seen)
print("\n=== RF: Test on SEEN attack types ===")
print("Accuracy:", accuracy_score(y_test_seen, y_pred_seen_rf))
print(classification_report(y_test_seen, y_pred_seen_rf, target_names=["bonafide", "spoof"]))

y_pred_unseen_rf = final_rf.predict(X_unseen)
print("\n=== RF: Test on UNSEEN attack types (A16, A09) ===")
print("Accuracy:", accuracy_score(y_unseen, y_pred_unseen_rf))
print(classification_report(y_unseen, y_pred_unseen_rf, target_names=["bonafide", "spoof"]))