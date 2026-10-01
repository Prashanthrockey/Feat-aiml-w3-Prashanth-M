import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

output_folder = "visualizations"
os.makedirs(output_folder, exist_ok=True)

cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 1. Baseline Model
baseline_rf = RandomForestClassifier(random_state=42)
baseline_rf.fit(X_train, y_train)
base_preds = baseline_rf.predict(X_test)

# 2. GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
grid_preds = grid_search.best_estimator_.predict(X_test)

# 3. RandomizedSearchCV
param_dist = {
    'n_estimators': [int(x) for x in np.linspace(20, 200, 10)],
    'max_depth': [3, 5, 10, 15, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='f1',
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)
rand_preds = random_search.best_estimator_.predict(X_test)

# Results Comparison
results = pd.DataFrame([
    {
        "Method": "Baseline (Default)",
        "Accuracy": round(accuracy_score(y_test, base_preds), 4),
        "Precision": round(precision_score(y_test, base_preds), 4),
        "Recall": round(recall_score(y_test, base_preds), 4),
        "F1-Score": round(f1_score(y_test, base_preds), 4)
    },
    {
        "Method": "GridSearchCV",
        "Accuracy": round(accuracy_score(y_test, grid_preds), 4),
        "Precision": round(precision_score(y_test, grid_preds), 4),
        "Recall": round(recall_score(y_test, grid_preds), 4),
        "F1-Score": round(f1_score(y_test, grid_preds), 4)
    },
    {
        "Method": "RandomizedSearchCV",
        "Accuracy": round(accuracy_score(y_test, rand_preds), 4),
        "Precision": round(precision_score(y_test, rand_preds), 4),
        "Recall": round(recall_score(y_test, rand_preds), 4),
        "F1-Score": round(f1_score(y_test, rand_preds), 4)
    }
])

# Visualization 1 - Table Comparison
fig, ax = plt.subplots(figsize=(9, 2.5))
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
plt.title("Hyperparameter Tuning Performance Comparison", fontsize=14, pad=10)
plt.savefig(os.path.join(output_folder, "day5_tuning_comparison.png"), bbox_inches='tight', dpi=300)
plt.show()

# Visualization 2 - Search Method CV Score Distribution
grid_scores = grid_search.cv_results_['mean_test_score']
random_scores = random_search.cv_results_['mean_test_score']

plt.figure(figsize=(10, 5))
plt.plot(grid_scores, label='GridSearchCV Fits', color='royalblue', alpha=0.7)
plt.plot(random_scores, label='RandomizedSearchCV Fits', color='darkorange', alpha=0.8, linestyle='--')
plt.axhline(y=grid_search.best_score_, color='blue', linestyle=':', label=f'Grid Best ({grid_search.best_score_:.4f})')
plt.axhline(y=random_search.best_score_, color='orange', linestyle=':', label=f'Random Best ({random_search.best_score_:.4f})')
plt.title("Cross-Validation F1-Scores across Iterations", fontsize=14)
plt.xlabel("Trial / Combination Index")
plt.ylabel("Mean CV F1-Score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "day5_cv_score_distribution.png"), bbox_inches='tight', dpi=300)
plt.show()

print(f"All Day 5 visualizations saved successfully in folder: '{output_folder}/'")