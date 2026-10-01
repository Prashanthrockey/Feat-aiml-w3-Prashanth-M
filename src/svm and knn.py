import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)

output_folder = "visualizations"
os.makedirs(output_folder, exist_ok=True)

cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_linear = SVC(kernel="linear", C=1.0, random_state=42)
svm_linear.fit(X_train_scaled, y_train)
svm_lin_preds = svm_linear.predict(X_test_scaled)

svm_rbf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
svm_rbf.fit(X_train_scaled, y_train)
svm_rbf_preds = svm_rbf.predict(X_test_scaled)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
knn_preds = knn.predict(X_test_scaled)

results = pd.DataFrame([
    {
        "Model": "SVM (Linear)",
        "Accuracy": round(accuracy_score(y_test, svm_lin_preds), 4),
        "Precision": round(precision_score(y_test, svm_lin_preds), 4),
        "Recall": round(recall_score(y_test, svm_lin_preds), 4),
        "F1-Score": round(f1_score(y_test, svm_lin_preds), 4)
    },
    {
        "Model": "SVM (RBF Kernel)",
        "Accuracy": round(accuracy_score(y_test, svm_rbf_preds), 4),
        "Precision": round(precision_score(y_test, svm_rbf_preds), 4),
        "Recall": round(recall_score(y_test, svm_rbf_preds), 4),
        "F1-Score": round(f1_score(y_test, svm_rbf_preds), 4)
    },
    {
        "Model": "KNN (K=5)",
        "Accuracy": round(accuracy_score(y_test, knn_preds), 4),
        "Precision": round(precision_score(y_test, knn_preds), 4),
        "Recall": round(recall_score(y_test, knn_preds), 4),
        "F1-Score": round(f1_score(y_test, knn_preds), 4)
    }
])

fig, ax = plt.subplots(figsize=(8, 2.5))
ax.axis('tight')
ax.axis('off')
table = ax.table(
    cellText=results.values, 
    colLabels=results.columns, 
    cellLoc='center', 
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.5)
plt.title("SVM vs KNN Model Comparison", fontsize=14, pad=10)
plt.savefig(os.path.join(output_folder, "day4_model_comparison.png"), bbox_inches='tight', dpi=300)
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

cm_lin = confusion_matrix(y_test, svm_lin_preds)
ConfusionMatrixDisplay(cm_lin, display_labels=cancer.target_names).plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("SVM (Linear)")

cm_rbf = confusion_matrix(y_test, svm_rbf_preds)
ConfusionMatrixDisplay(cm_rbf, display_labels=cancer.target_names).plot(ax=axes[1], cmap="Purples", colorbar=False)
axes[1].set_title("SVM (RBF)")

cm_knn = confusion_matrix(y_test, knn_preds)
ConfusionMatrixDisplay(cm_knn, display_labels=cancer.target_names).plot(ax=axes[2], cmap="Greens", colorbar=False)
axes[2].set_title("KNN (K=5)")

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "day4_confusion_matrices.png"), bbox_inches='tight', dpi=300)
plt.show()

X_2d = X_train_scaled[:, :2]
y_2d = np.array(y_train)

svm_2d = SVC(kernel="rbf", C=1.0).fit(X_2d, y_2d)
knn_2d = KNeighborsClassifier(n_neighbors=5).fit(X_2d, y_2d)

x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

Z_svm = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
axes[0].contourf(xx, yy, Z_svm, alpha=0.3, cmap=plt.cm.coolwarm)
axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.coolwarm, edgecolors="k")
axes[0].set_title("SVM (RBF) Decision Boundary")

Z_knn = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
axes[1].contourf(xx, yy, Z_knn, alpha=0.3, cmap=plt.cm.coolwarm)
axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.coolwarm, edgecolors="k")
axes[1].set_title("KNN (K=5) Decision Boundary")

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "day4_decision_boundaries.png"), bbox_inches='tight', dpi=300)
plt.show()

print(f"All Day 4 visualizations saved successfully in folder: '{output_folder}/'")