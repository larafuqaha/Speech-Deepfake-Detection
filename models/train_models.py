import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)

os.makedirs("results", exist_ok=True)

# ------- Load data -------
X = pd.read_csv("data/features.csv").values
y = pd.read_csv("data/labels.csv")["label"].values

# =========================================
# Train/test split 
# =========================================
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit scaler on trainval ONLY, apply to test — avoids data leakage
scaler = StandardScaler()
X_trainval = scaler.fit_transform(X_trainval)
X_test = scaler.transform(X_test)

# =========================================
# SVM: tune with GridSearchCV 
# =========================================
svm_grid = GridSearchCV(
    SVC(kernel="rbf", probability=True),
    param_grid={"C": [0.1, 1, 10]},
    cv=5,
    scoring="accuracy"
)
svm_grid.fit(X_trainval, y_trainval)
print("Best SVM params:", svm_grid.best_params_, "| CV accuracy:", round(svm_grid.best_score_, 3))
final_svm = svm_grid.best_estimator_

# =========================================
# Random Forest: tune with GridSearchCV 
# =========================================
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20]},
    cv=5,
    scoring="accuracy"
)
rf_grid.fit(X_trainval, y_trainval)
print("Best RF params:", rf_grid.best_params_, "| CV accuracy:", round(rf_grid.best_score_, 3))
final_rf = rf_grid.best_estimator_

# =========================================
# Evaluate both on the TEST set (once each) 
# =========================================
y_pred_svm = final_svm.predict(X_test)
print("\n--- SVM Test Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm, target_names=["bonafide", "spoof"]))
print(confusion_matrix(y_test, y_pred_svm))

y_pred_rf = final_rf.predict(X_test)
print("\n--- Random Forest Test Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, target_names=["bonafide", "spoof"]))
print(confusion_matrix(y_test, y_pred_rf))

# =========================================
# Plot: confusion matrices 
# =========================================
def plot_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["bonafide", "spoof"], yticklabels=["bonafide", "spoof"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"results/{filename}")
    plt.close()

plot_confusion(y_test, y_pred_svm, "SVM Confusion Matrix", "svm_confusion.png")
plot_confusion(y_test, y_pred_rf, "Random Forest Confusion Matrix", "rf_confusion.png")

# =========================================
# Plot: ROC curve comparison 
# =========================================
plt.figure(figsize=(6, 5))
for name, model in [("SVM", final_svm), ("Random Forest", final_rf)]:
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — SVM vs Random Forest")
plt.legend()
plt.tight_layout()
plt.savefig("results/roc_comparison.png")
plt.close()

print("\nPlots saved to results/: svm_confusion.png, rf_confusion.png, roc_comparison.png")