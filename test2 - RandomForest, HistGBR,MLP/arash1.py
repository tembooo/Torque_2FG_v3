# compare_models.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RepeatedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
import joblib

import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ------------------------------
# Paths
# ------------------------------
BASE_DIR = r"C:\12.LUT\00.Termic Cources\0.Finished\Arash\4.4th project\M1.Torque_2FG_v3"
FILE_XLSX = os.path.join(BASE_DIR, "1.Torque_2FG_v3.xlsx")

RESULTS_JSON   = os.path.join(BASE_DIR, "compare_results.json")
PRED_XLSX_RF   = os.path.join(BASE_DIR, "predictions_randomforest.xlsx")
PRED_XLSX_HGB  = os.path.join(BASE_DIR, "predictions_histgbr.xlsx")
PRED_XLSX_MLP  = os.path.join(BASE_DIR, "predictions_mlp.xlsx")
PARITY_PNG     = os.path.join(BASE_DIR, "parity_best.png")
RESID_PNG      = os.path.join(BASE_DIR, "residuals_best.png")
F9_PNG         = os.path.join(BASE_DIR, "actual_vs_pred_F9_best.png")
FI_PNG         = os.path.join(BASE_DIR, "feature_importance_best.png")
SCALER_PATH    = os.path.join(BASE_DIR, "compare_scaler.joblib")
BEST_MODEL_BIN = os.path.join(BASE_DIR, "best_model.joblib")

# ------------------------------
# Load data
# ------------------------------
if not os.path.exists(FILE_XLSX):
    raise FileNotFoundError(f"Excel not found: {FILE_XLSX}")

df = pd.read_excel(FILE_XLSX)
cols_expected = [f"F{i}" for i in range(1, 10)] + ["Torque"]
missing = [c for c in cols_expected if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

X = df[[f"F{i}" for i in range(1, 10)]].copy()
y = df["Torque"].values

# ------------------------------
# Split
# ------------------------------
X_train, X_hold, y_train, y_hold = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# ------------------------------
# Models
# ------------------------------
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=4,
    random_state=42,
    n_jobs=-1
)

hgb = HistGradientBoostingRegressor(
    max_depth=None,
    learning_rate=0.05,
    max_iter=500,
    l2_regularization=1e-3,
    random_state=42
)

def build_mlp(input_dim, units1=128, units2=64, lr=1e-3):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(units1, activation="relu"),
        Dense(units2, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return model

# ------------------------------
# Train/evaluate helper
# ------------------------------
def evaluate_and_report(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    return {"MSE": float(mse), "RMSE": float(rmse), "MAE": float(mae), "R2": float(r2)}

# ------------------------------
# Cross-Validation for tree models
# ------------------------------
cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)

def cv_scores(model, X, y):
    sc = cross_validate(
        model, X, y,
        scoring=("r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"),
        cv=cv, n_jobs=-1, return_train_score=False
    )
    return {
        "CV_R2_mean": float(np.mean(sc["test_r2"])),
        "CV_R2_std": float(np.std(sc["test_r2"])),
        "CV_MAE_mean": float(-np.mean(sc["test_neg_mean_absolute_error"])),
        "CV_RMSE_mean": float(-np.mean(sc["test_neg_root_mean_squared_error"])),
    }

# ------------------------------
# RandomForest
# ------------------------------
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_hold)
res_rf = evaluate_and_report("RandomForest", y_hold, y_pred_rf)
res_rf.update(cv_scores(rf, X_train, y_train))
pred_rf = df.copy()
pred_rf["Torque_pred"] = rf.predict(X)
pred_rf.to_excel(PRED_XLSX_RF, index=False)

# ------------------------------
# HistGradientBoosting
# ------------------------------
hgb.fit(X_train, y_train)
y_pred_hgb = hgb.predict(X_hold)
res_hgb = evaluate_and_report("HistGBR", y_hold, y_pred_hgb)
res_hgb.update(cv_scores(hgb, X_train, y_train))
pred_hgb = df.copy()
pred_hgb["Torque_pred"] = hgb.predict(X)
pred_hgb.to_excel(PRED_XLSX_HGB, index=False)

# ------------------------------
# MLP (scaled)
# ------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_hold_s  = scaler.transform(X_hold)
X_all_s   = scaler.transform(X)

mlp = build_mlp(X_train.shape[1], units1=128, units2=64, lr=1e-3)
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, min_lr=1e-6),
]
mlp.fit(
    X_train_s, y_train,
    validation_split=0.1,
    epochs=200,
    batch_size=32,
    callbacks=callbacks,
    verbose=2
)
y_pred_mlp = mlp.predict(X_hold_s, verbose=0).flatten()
res_mlp = evaluate_and_report("MLP", y_hold, y_pred_mlp)

pred_mlp = df.copy()
pred_mlp["Torque_pred"] = mlp.predict(X_all_s, verbose=0).flatten()
pred_mlp.to_excel(PRED_XLSX_MLP, index=False)
joblib.dump(scaler, SCALER_PATH)

# ------------------------------
# Compare and pick best (by RMSE holdout)
# ------------------------------
results = {
    "RandomForest": res_rf,
    "HistGBR": res_hgb,
    "MLP": res_mlp
}

best_name = min(results.keys(), key=lambda k: results[k]["RMSE"])
print("Results:")
for k, v in results.items():
    print(k, v)
print("Best by holdout RMSE:", best_name, results[best_name])

with open(RESULTS_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# Save best model binary (for sklearn ones); for MLP save SavedModel
if best_name == "RandomForest":
    joblib.dump(rf, BEST_MODEL_BIN)
    best_pred = pred_rf["Torque_pred"].values
    best_model_type = "sklearn"
elif best_name == "HistGBR":
    joblib.dump(hgb, BEST_MODEL_BIN)
    best_pred = pred_hgb["Torque_pred"].values
    best_model_type = "sklearn"
else:
    mlp.save(os.path.join(BASE_DIR, "best_mlp_model.h5"))
    best_pred = pred_mlp["Torque_pred"].values
    best_model_type = "keras"

# ------------------------------
# Plots for best model
# ------------------------------
# Parity
plt.figure(figsize=(6,6))
if best_name == "RandomForest":
    y_pred_hold = y_pred_rf
elif best_name == "HistGBR":
    y_pred_hold = y_pred_hgb
else:
    y_pred_hold = y_pred_mlp

mn, mx = min(y_hold.min(), y_pred_hold.min()), max(y_hold.max(), y_pred_hold.max())
plt.scatter(y_hold, y_pred_hold, alpha=0.6)
plt.plot([mn, mx], [mn, mx], "--")
plt.xlabel("Actual Torque")
plt.ylabel("Predicted Torque")
plt.title(f"Parity Plot (Holdout) | {best_name}")
plt.grid(True); plt.tight_layout()
plt.savefig(PARITY_PNG, dpi=150); plt.close()

# Residuals
resid = y_pred_hold - y_hold
plt.figure(figsize=(10,4))
plt.plot(resid, marker="o", linestyle="None", alpha=0.6)
plt.axhline(0, color="k", linestyle="--")
plt.title(f"Residuals (Holdout) | {best_name}")
plt.xlabel("Sample index")
plt.ylabel("Residual")
plt.grid(True); plt.tight_layout()
plt.savefig(RESID_PNG, dpi=150); plt.close()

# Actual vs Pred by F9
plt.figure(figsize=(10,6))
plt.scatter(df["F9"], df["Torque"], alpha=0.6, label="Actual")
plt.scatter(df["F9"], best_pred, alpha=0.6, label="Predicted")
plt.xlabel("F9")
plt.ylabel("Torque")
plt.title(f"Actual vs Predicted Torque by F9 (Best: {best_name})")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(F9_PNG, dpi=150); plt.close()

# Feature importance for tree models
# --- جایگزینِ بلوک "Feature importance for tree models" ---

from sklearn.inspection import permutation_importance

FI_PNG = os.path.join(BASE_DIR, "feature_importance_best.png")
FI_CSV = os.path.join(BASE_DIR, "feature_importance_best.csv")

if best_name in ("RandomForest", "HistGBR"):
    if best_name == "RandomForest":
        # 1) RF: از feature_importances_ استفاده کن
        importances = rf.feature_importances_
        names = np.array(X.columns)
        order = np.argsort(importances)[::-1]
        imp_vals = importances[order]
        imp_names = names[order]

    else:
        # 2) HistGBR: Permutation Importance
        # روی holdout حساب می‌کنیم تا unbiased باشد
        r = permutation_importance(
            hgb, X_hold, y_hold,
            n_repeats=10, random_state=42, n_jobs=-1,
            scoring="neg_mean_squared_error"  # می‌تونی "r2" یا "neg_mean_absolute_error" هم بگذاری
        )
        importances = r.importances_mean
        names = np.array(X.columns)
        order = np.argsort(importances)[::-1]
        imp_vals = importances[order]
        imp_names = names[order]

    # ذخیره CSV
    pd.DataFrame({"feature": imp_names, "importance": imp_vals}).to_csv(FI_CSV, index=False)

    # رسم نمودار
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(imp_vals)), imp_vals)
    plt.xticks(range(len(imp_vals)), imp_names, rotation=45, ha="right")
    plt.title(f"Feature Importances ({best_name})")
    plt.tight_layout()
    plt.savefig(FI_PNG, dpi=150); plt.close()

    print("Saved feature importance:", FI_PNG, "and", FI_CSV)

