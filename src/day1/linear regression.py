import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create synthetic housing dataset locally (offline)
X_raw, y = make_regression(n_samples=1000, n_features=8, noise=15.0, random_state=42)
feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
X = pd.DataFrame(X_raw, columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def get_metrics(model, name, x_train, x_test):
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    return {
        "Model": name,
        "MSE": round(mean_squared_error(y_test, preds), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 4),
        "MAE": round(mean_absolute_error(y_test, preds), 4),
        "R2 Score": round(r2_score(y_test, preds), 4)
    }, preds, model

lr_metrics, lr_preds, lr_model = get_metrics(LinearRegression(), "Linear Regression", X_train, X_test)
ridge_metrics, _, _ = get_metrics(Ridge(alpha=1.0), "Ridge Regression", X_train_scaled, X_test_scaled)
lasso_metrics, _, _ = get_metrics(Lasso(alpha=0.01), "Lasso Regression", X_train_scaled, X_test_scaled)

print("--- Model Coefficients ---")
for feature, coef in zip(feature_names, lr_model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {lr_model.intercept_:.4f}\n")

print("--- Model Comparison ---")
results = pd.DataFrame([lr_metrics, ridge_metrics, lasso_metrics])
print(results.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_test, lr_preds, alpha=0.3, color="blue")
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
axes[0].set_title("Predicted vs Actual")

residuals = y_test - lr_preds
axes[1].scatter(lr_preds, residuals, alpha=0.3, color="purple")
axes[1].axhline(y=0, color="r", linestyle="--")
axes[1].set_title("Residuals")

plt.tight_layout()
plt.show()