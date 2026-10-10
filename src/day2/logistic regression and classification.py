import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)

cancer = load_breast_cancer()
X_bin = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y_bin = cancer.target

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_bin, y_bin, test_size=0.2, random_state=42, stratify=y_bin
)

scaler_b = StandardScaler()
X_train_b_scaled = scaler_b.fit_transform(X_train_b)
X_test_b_scaled = scaler_b.transform(X_test_b)

log_reg_bin = LogisticRegression(max_iter=10000, random_state=42)
log_reg_bin.fit(X_train_b_scaled, y_train_b)

print("Intercept:", round(log_reg_bin.intercept_[0], 4))
print("Top Feature Coefficients:")
for feat, coef in zip(cancer.feature_names[:5], log_reg_bin.coef_[0][:5]):
    print(f"{feat}: {coef:.4f}")
print("\n")

y_pred_b = log_reg_bin.predict(X_test_b_scaled)
y_proba_b = log_reg_bin.predict_proba(X_test_b_scaled)[:, 1]

print("Binary Classification Metrics:")
print("Accuracy :", round(accuracy_score(y_test_b, y_pred_b), 4))
print("Precision:", round(precision_score(y_test_b, y_pred_b), 4))
print("Recall   :", round(recall_score(y_test_b, y_pred_b), 4))
print("F1-Score :", round(f1_score(y_test_b, y_pred_b), 4))
print("ROC-AUC  :", round(roc_auc_score(y_test_b, y_proba_b), 4))
print("\n")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm = confusion_matrix(y_test_b, y_pred_b)
ConfusionMatrixDisplay(cm, display_labels=cancer.target_names).plot(ax=axes[0], cmap="Blues")
axes[0].set_title("Confusion Matrix")

fpr, tpr, _ = roc_curve(y_test_b, y_proba_b)
auc_val = roc_auc_score(y_test_b, y_proba_b)
axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_val:.2f})")
axes[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve")
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.show()

X_2d = X_train_b_scaled[:, :2]
y_2d = np.array(y_train_b)

model_2d = LogisticRegression()
model_2d.fit(X_2d, y_2d)

x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Spectral)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.Spectral, edgecolors="k")
plt.xlabel(cancer.feature_names[0])
plt.ylabel(cancer.feature_names[1])
plt.title("2D Decision Boundary")
plt.show()

iris = load_iris()
X_m, y_m = iris.data, iris.target

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_m, y_m, test_size=0.2, random_state=42, stratify=y_m
)

scaler_m = StandardScaler()
X_train_m_scaled = scaler_m.fit_transform(X_train_m)
X_test_m_scaled = scaler_m.transform(X_test_m)

ovr_model = OneVsRestClassifier(LogisticRegression(random_state=42))
ovr_model.fit(X_train_m_scaled, y_train_m)
ovr_preds = ovr_model.predict(X_test_m_scaled)
ovr_proba = ovr_model.predict_proba(X_test_m_scaled)

softmax_model = LogisticRegression(solver="lbfgs", random_state=42)
softmax_model.fit(X_train_m_scaled, y_train_m)
softmax_preds = softmax_model.predict(X_test_m_scaled)
softmax_proba = softmax_model.predict_proba(X_test_m_scaled)

multi_results = pd.DataFrame([
    {
        "Strategy": "One-vs-Rest (OvR)",
        "Accuracy": round(accuracy_score(y_test_m, ovr_preds), 4),
        "Precision (Macro)": round(precision_score(y_test_m, ovr_preds, average="macro"), 4),
        "Recall (Macro)": round(recall_score(y_test_m, ovr_preds, average="macro"), 4),
        "F1-Score (Macro)": round(f1_score(y_test_m, ovr_preds, average="macro"), 4),
        "ROC-AUC (OVO)": round(roc_auc_score(y_test_m, ovr_proba, multi_class="ovo"), 4)
    },
    {
        "Strategy": "Softmax (Multinomial)",
        "Accuracy": round(accuracy_score(y_test_m, softmax_preds), 4),
        "Precision (Macro)": round(precision_score(y_test_m, softmax_preds, average="macro"), 4),
        "Recall (Macro)": round(recall_score(y_test_m, softmax_preds, average="macro"), 4),
        "F1-Score (Macro)": round(f1_score(y_test_m, softmax_preds, average="macro"), 4),
        "ROC-AUC (OVO)": round(roc_auc_score(y_test_m, softmax_proba, multi_class="ovo"), 4)
    }
])

print("Multi-Class Strategy Comparison:")
print(multi_results.to_string(index=False))