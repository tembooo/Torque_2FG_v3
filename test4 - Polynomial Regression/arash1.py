import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataclasses import dataclass
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import joblib
import optuna

# ------------------------------
# Paths
# ------------------------------
BASE_DIR = r"C:\12.LUT\00.Termic Cources\0.Finished\Arash\4.4th project\M1.Torque_2FG_v3"
FILE_XLSX = os.path.join(BASE_DIR, "1.Torque_2FG_v3.xlsx")

PRED_XLSX        = os.path.join(BASE_DIR, "predictions_all_models.xlsx")
RESULTS_JSON     = os.path.join(BASE_DIR, "results_all_models.json")
BEST_MODEL_RAW   = os.path.join(BASE_DIR, "best_histgbr_optuna.joblib")
BEST_MODEL_RULE  = os.path.join(BASE_DIR, "best_histgbr_with_rule.joblib")
PARITY_PNG       = os.path.join(BASE_DIR, "parity_histgbr_optuna.png")
RESID_PNG        = os.path.join(BASE_DIR, "residuals_histgbr_optuna.png")
F9_PNG           = os.path.join(BASE_DIR, "actual_vs_pred_F9.png")
FI_PNG           = os.path.join(BASE_DIR, "feature_importance.png")
FI_CSV           = os.path.join(BASE_DIR, "feature_importance.csv")

# ------------------------------
# Reproducibility
# ------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ------------------------------
# Load data
# ------------------------------
if not os.path.exists(FILE_XLSX):
    raise FileNotFoundError(f"Excel not found: {FILE_XLSX}")

df = pd.read_excel(FILE_XLSX)
input_cols = [f"F{i}" for i in range(1, 10)]
target_col = "Torque"
missing = [c for c in input_cols + [target_col] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

X = df[input_cols].copy()
y = df[target_col].values

# ------------------------------
# Split holdout
# ------------------------------
X_train, X_hold, y_train, y_hold = train_test_split(
    X, y, test_size=0.2, random_state=SEED, shuffle=True
)

# ------------------------------
# Physics rule wrapper: if F9==0 -> torque_pred=0
# ------------------------------
@dataclass
class ZeroCurrentZeroTorqueWrapper(BaseEstimator, RegressorMixin):
    base_estimator: object
    current_feature: str = "F9"
    zero_current_weight: float = 5.0

    def fit(self, X, y):
        self.feature_names_in_ = list(X.columns) if hasattr(X, "columns") else [f"F{i}" for i in range(1, X.shape[1]+1)]
        self.current_idx_ = self.feature_names_in_.index(self.current_feature)
        x_np = X.values if hasattr(X, "values") else np.asarray(X)
        w = np.ones(len(y), dtype=float)
        w[x_np[:, self.current_idx_] == 0] = self.zero_current_weight
        self.model_ = clone(self.base_estimator)
        self.model_.fit(X, y, sample_weight=w)
        return self

    def predict(self, X):
        x_np = X.values if hasattr(X, "values") else np.asarray(X)
        preds = self.model_.predict(X)
        mask_zero = (x_np[:, self.current_idx_] == 0)
        preds = np.asarray(preds, dtype=float)
        preds[mask_zero] = 0.0
        return preds

# ------------------------------
# Optuna objective (HGBR)
# ------------------------------
cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEED)

def objective(trial: optuna.Trial) -> float:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_iter": trial.suggest_int("max_iter", 200, 1500, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 80),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 1e-1, log=True),
        "max_bins": trial.suggest_int("max_bins", 64, 255),
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": trial.suggest_int("n_iter_no_change", 10, 50),
        "random_state": SEED,
    }
    base = HistGradientBoostingRegressor(**params)
    model = ZeroCurrentZeroTorqueWrapper(base_estimator=base, current_feature="F9", zero_current_weight=5.0)

    rmses = []
    for tr_idx, te_idx in cv.split(X_train, y_train):
        X_tr, X_te = X_train.iloc[tr_idx], X_train.iloc[te_idx]
        y_tr, y_te = y_train[tr_idx], y_train[te_idx]
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_te)
        rmse = mean_squared_error(y_te, y_hat, squared=False)
        rmses.append(rmse)

    return float(np.mean(rmses))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=40, show_progress_bar=False)
best_params = study.best_trial.params
print("Best params:", best_params)
print("Best CV RMSE:", study.best_value)

# ------------------------------
# Train final models
# ------------------------------
best_base = HistGradientBoostingRegressor(
    **{**best_params, "early_stopping": True, "validation_fraction": 0.1, "random_state": SEED}
)
final_model_raw = best_base.fit(X_train, y_train)
final_model = ZeroCurrentZeroTorqueWrapper(
    base_estimator=best_base,
    current_feature="F9",
    zero_current_weight=5.0
).fit(X_train, y_train)

# Linear regression (baseline continuous)
lin_reg = LinearRegression().fit(X_train, y_train)

# Polynomial regression (degree=2 for smoother curve)
poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(X_train, y_train)

# ------------------------------
# Evaluate
# ------------------------------
def report(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    return {"MSE": float(mse), "RMSE": float(rmse), "MAE": float(mae), "R2": float(r2)}

metrics = {
    "HistGBR_raw": report(y_hold, final_model_raw.predict(X_hold)),
    "HistGBR_rule": report(y_hold, final_model.predict(X_hold)),
    "Linear": report(y_hold, lin_reg.predict(X_hold)),
    "Polynomial": report(y_hold, poly_model.predict(X_hold)),
}
print("Holdout metrics:", json.dumps(metrics, indent=2))

# ------------------------------
# Save predictions
# ------------------------------
y_all_pred_histgbr = final_model.predict(X)
y_all_pred_linear = lin_reg.predict(X)
y_all_pred_poly = poly_model.predict(X)

out = df.copy()
out["Torque_pred_HistGBR"] = y_all_pred_histgbr
out["Torque_pred_Linear"] = y_all_pred_linear
out["Torque_pred_Poly"] = y_all_pred_poly
out.to_excel(PRED_XLSX, index=False)
print("Saved predictions:", PRED_XLSX)

# ------------------------------
# Save models
# ------------------------------
joblib.dump(final_model_raw, BEST_MODEL_RAW)
joblib.dump(final_model, BEST_MODEL_RULE)
joblib.dump(lin_reg, os.path.join(BASE_DIR, "linear_reg.joblib"))
joblib.dump(poly_model, os.path.join(BASE_DIR, "poly_reg.joblib"))

# ------------------------------
# Plots
# ------------------------------
# Actual vs predicted (F9)
plt.figure(figsize=(10,6))
plt.scatter(df["F9"], df["Torque"], alpha=0.6, label="Actual")
plt.scatter(df["F9"], out["Torque_pred_HistGBR"], alpha=0.6, label="HistGBR")
plt.plot(df["F9"], out["Torque_pred_Linear"], "g--", label="Linear")
plt.plot(df["F9"], out["Torque_pred_Poly"], "r-", label="Polynomial")
plt.xlabel("F9")
plt.ylabel("Torque")
plt.title("Actual vs Predicted Torque by F9 (Different Models)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(F9_PNG, dpi=150); plt.close()

# Parity plot for HistGBR
plt.figure(figsize=(6,6))
mn, mx = min(y_hold.min(), y_all_pred_histgbr.min()), max(y_hold.max(), y_all_pred_histgbr.max())
plt.scatter(y_hold, final_model.predict(X_hold), alpha=0.6)
plt.plot([mn, mx], [mn, mx], "--")
plt.xlabel("Actual Torque")
plt.ylabel("Predicted Torque")
plt.title("Parity Plot (HistGBR+Optuna+Rule)")
plt.grid(True); plt.tight_layout()
plt.savefig(PARITY_PNG, dpi=150); plt.close()

# Residuals (HistGBR)
resid = final_model.predict(X_hold) - y_hold
plt.figure(figsize=(10,4))
plt.plot(resid, marker="o", linestyle="None", alpha=0.6)
plt.axhline(0, color="k", linestyle="--")
plt.title("Residuals (Holdout) | HistGBR+Optuna+Rule")
plt.xlabel("Sample index")
plt.ylabel("Residual")
plt.grid(True); plt.tight_layout()
plt.savefig(RESID_PNG, dpi=150); plt.close()

# Feature importance
r = permutation_importance(
    final_model_raw, X_hold, y_hold,
    n_repeats=10, random_state=SEED, n_jobs=-1, scoring="neg_mean_squared_error"
)
importances = r.importances_mean
names = np.array(X.columns)
order = np.argsort(importances)[::-1]
imp_vals = importances[order]
imp_names = names[order]

pd.DataFrame({"feature": imp_names, "importance": imp_vals}).to_csv(FI_CSV, index=False)
plt.figure(figsize=(8,5))
plt.bar(range(len(imp_vals)), imp_vals)
plt.xticks(range(len(imp_vals)), imp_names, rotation=45, ha="right")
plt.title("Feature Importances (HistGBR+Optuna)")
plt.tight_layout()
plt.savefig(FI_PNG, dpi=150); plt.close()

# ------------------------------
# Save results json
# ------------------------------
results = {
    "best_params_histgbr": best_params,
    "metrics": metrics,
    "paths": {
        "predictions": PRED_XLSX,
        "model_histgbr_raw": BEST_MODEL_RAW,
        "model_histgbr_rule": BEST_MODEL_RULE,
        "linear": "linear_reg.joblib",
        "poly": "poly_reg.joblib",
        "parity_png": PARITY_PNG,
        "residuals_png": RESID_PNG,
        "f9_png": F9_PNG,
        "fi_png": FI_PNG,
        "fi_csv": FI_CSV
    }
}
with open(RESULTS_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Done. Results saved to:", RESULTS_JSON)
