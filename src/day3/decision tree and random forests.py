import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

results = pd.DataFrame([
    {
        "Model": "Decision Tree",
        "Accuracy": round(accuracy_score(y_test, dt_preds), 4),
        "Precision": round(precision_score(y_test, dt_preds, average="macro"), 4),
        "Recall": round(recall_score(y_test, dt_preds, average="macro"), 4),
        "F1-Score": round(f1_score(y_test, dt_preds, average="macro"), 4)
    },
    {
        "Model": "Random Forest",
        "Accuracy": round(accuracy_score(y_test, rf_preds), 4),
        "Precision": round(precision_score(y_test, rf_preds, average="macro"), 4),
        "Recall": round(recall_score(y_test, rf_preds, average="macro"), 4),
        "F1-Score": round(f1_score(y_test, rf_preds, average="macro"), 4)
    }
])

print("Model Performance Comparison:")
print(results.to_string(index=False))
print("\n")

plt.figure(figsize=(14, 7))
plot_tree(
    dt_model, 
    feature_names=wine.feature_names, 
    class_names=wine.target_names, 
    filled=True, 
    rounded=True
)
plt.title("Decision Tree Structure (Max Depth = 3)")
plt.tight_layout()
plt.show()

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.bar(range(X.shape[1]), importances[indices], align="center", color="teal")
plt.xticks(range(X.shape[1]), [wine.feature_names[i] for i in indices], rotation=45, ha="right")
plt.title("Random Forest - Feature Importances")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()