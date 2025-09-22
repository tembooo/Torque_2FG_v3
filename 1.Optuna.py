# -*- coding: utf-8 -*-
import os, json, random
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
try:
    from sklearn.metrics import root_mean_squared_error
except Exception:
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)
import joblib, optuna

# ---------------- Paths (everything beside this script) ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

FILE_XLSX    = os.path.join(BASE_DIR, "2.Ld-Lq_2FG_v3.xlsx")
PRED_XLSX    = os.path.join(BASE_DIR, "predictions_mlp_optuna_LdLq.xlsx")
RESULTS_JSON = os.path.join(BASE_DIR, "mlp_optuna_results_LdLq.json")
BEST_MODEL   = os.path.join(BASE_DIR, "best_mlp_model_LdLq.keras")  # Keras native format
SCALER_X_JOB = os.path.join(BASE_DIR, "scaler_X_LdLq.joblib")
SCALER_Y_JOB = os.path.join(BASE_DIR, "scaler_y_LdLq.joblib")
PARITY_PNG   = os.path.join(BASE_DIR, "parity_mlp_LdLq.png")
RESID_PNG    = os.path.join(BASE_DIR, "residuals_mlp_LdLq.png")
F9_PNG       = os.path.join(BASE_DIR, "actual_vs_pred_F9_mlp_LdLq.png")

# New: Optuna exports beside script
TRIALS_XLSX  = os.path.join(BASE_DIR, "all_trials.xlsx")
TRIALS_PNG   = os.path.join(BASE_DIR, "validation_loss_by_trial.png")
OPTUNA_DB    = os.path.join(BASE_DIR, "optuna_trials.db")

# ---------------- Reproducibility ----------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ---------------- Load & prep ----------------
df = pd.read_excel(FILE_XLSX)

feature_cols = sorted(
    [c for c in df.columns if c.startswith("F") and c[1:].isdigit()],
    key=lambda x: int(x[1:])
)

target_col = "Ld-Lq"
assert target_col in df.columns, f"Target '{target_col}' not found."
df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

X = df[feature_cols].to_numpy(np.float64)
y = df[[target_col]].to_numpy(np.float64)

scaler_X = StandardScaler(); scaler_y = StandardScaler()
X = scaler_X.fit_transform(X); y = scaler_y.fit_transform(y)

X_train, X_hold, y_train, y_hold = train_test_split(
    X, y, test_size=0.2, random_state=SEED, shuffle=True
)

# ---------------- Model ----------------
def build_mlp(n_layers, n_units, dropout_rate, input_dim, lr):
    m = keras.Sequential([layers.Input((input_dim,))])
    for _ in range(n_layers):
        m.add(layers.Dense(n_units, activation="relu"))
        m.add(layers.Dropout(dropout_rate))
    m.add(layers.Dense(1))
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return m

# ---------------- Optuna ----------------
def objective(trial):
    n_layers   = trial.suggest_int("n_layers", 1, 2)
    n_units    = trial.suggest_int("n_units", 32, 96, step=32)     # {32, 64, 96}
    dropout    = trial.suggest_float("dropout_rate", 0.0, 0.4)
    lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    model = build_mlp(n_layers, n_units, dropout, X_train.shape[1], lr)
    es = callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    rlrop = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6)

    hist = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=batch_size,
        verbose=2,
        callbacks=[es, rlrop]
    )
    return float(np.min(hist.history["val_loss"]))

# Persistent storage so trials are saved to disk (in the same folder)
storage_uri = f"sqlite:///{OPTUNA_DB.replace(os.sep, '/')}"
study = optuna.create_study(
    direction="minimize",
    study_name="mlp_ld_lq_opt",
    storage=storage_uri,
    load_if_exists=True
)
study.optimize(objective, n_trials=40, show_progress_bar=False)

best = study.best_trial.params
print("Best params:", best)

# ---- Export all trials to Excel + Plot
records = []
for t in study.trials:
    if t.value is None:
        continue
    records.append({
        "trial": t.number,
        "val_loss": t.value,
        "n_layers": t.params.get("n_layers"),
        "n_units": t.params.get("n_units"),
        "dropout_rate": t.params.get("dropout_rate"),
        "lr": t.params.get("lr"),
        "batch_size": t.params.get("batch_size"),
        "state": str(t.state),
    })

df_trials = pd.DataFrame(records).sort_values("trial").reset_index(drop=True)
if not df_trials.empty:
    df_trials.to_excel(TRIALS_XLSX, index=False)

    plt.figure(figsize=(10,7))
    plt.scatter(df_trials["trial"], df_trials["val_loss"], marker="x")
    best_idx = df_trials["val_loss"].idxmin()
    plt.scatter(df_trials.loc[best_idx, "trial"], df_trials.loc[best_idx, "val_loss"],
                marker="x", s=120, linewidths=2)
    plt.title("Validation Loss by Trial")
    plt.xlabel("trial (number)")
    plt.ylabel("val_loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(TRIALS_PNG, dpi=150)
    plt.close()

    print("Saved trials Excel:", TRIALS_XLSX)
    print("Saved trials plot :", TRIALS_PNG)
else:
    print("No completed trials to export.")

# ---------------- Final train with best params ----------------
model = build_mlp(best["n_layers"], best["n_units"], best["dropout_rate"],
                  X_train.shape[1], best["lr"])
es = callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
rlrop = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6)
model.fit(
    X_train, y_train,
    validation_data=(X_hold, y_hold),
    epochs=500,
    batch_size=best["batch_size"],
    callbacks=[es, rlrop],
    verbose=2
)

model.save(BEST_MODEL)
joblib.dump(scaler_X, SCALER_X_JOB); joblib.dump(scaler_y, SCALER_Y_JOB)
print("Saved model:", BEST_MODEL)

# ---------------- Evaluate ----------------
y_pred = scaler_y.inverse_transform(model.predict(X_hold))
y_true = scaler_y.inverse_transform(y_hold)
mse = mean_squared_error(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2  = r2_score(y_true, y_pred)
metrics = {"MSE": float(mse), "RMSE": float(rmse), "MAE": float(mae), "R2": float(r2)}
print("Holdout metrics:", metrics)

# ---------------- Save predictions all rows ----------------
y_all_pred = scaler_y.inverse_transform(model.predict(X)).ravel()
out = df.copy(); out["Ld-Lq_pred_MLP"] = y_all_pred
out.to_excel(PRED_XLSX, index=False)
print("Saved predictions:", PRED_XLSX)

# ---------------- Plots ----------------
# Parity
plt.figure(figsize=(6,6))
mn, mx = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
plt.scatter(y_true, y_pred, alpha=0.6); plt.plot([mn, mx], [mn, mx], "--")
plt.xlabel("Actual Ld-Lq"); plt.ylabel("Predicted Ld-Lq"); plt.title("Parity (MLP)"); plt.grid(True); plt.tight_layout()
plt.savefig(PARITY_PNG, dpi=150); plt.close()

# Residuals
resid = y_pred.flatten() - y_true.flatten()
plt.figure(figsize=(10,4))
plt.plot(resid, marker="o", linestyle="None", alpha=0.6); plt.axhline(0, linestyle="--")
plt.title("Residuals (Holdout) | Ld-Lq"); plt.xlabel("Index"); plt.ylabel("Residual"); plt.grid(True); plt.tight_layout()
plt.savefig(RESID_PNG, dpi=150); plt.close()

# Actual vs Pred by F9 (if exists)
if "F9" in df.columns:
    plt.figure(figsize=(10,6))
    plt.scatter(df["F9"], df[target_col], alpha=0.6, label="Actual")
    plt.scatter(df["F9"], out["Ld-Lq_pred_MLP"], alpha=0.6, label="Predicted")
    plt.xlabel("F9"); plt.ylabel("Ld-Lq"); plt.title("Actual vs Predicted by F9"); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(F9_PNG, dpi=150); plt.close()

# ---------------- Results JSON ----------------
with open(RESULTS_JSON, "w", encoding="utf-8") as f:
    json.dump({
        "best_params": best,
        "metrics": metrics,
        "columns": {"features": feature_cols, "target": target_col},
        "paths": {
            "predictions": PRED_XLSX, "model": BEST_MODEL,
            "scaler_X": SCALER_X_JOB, "scaler_y": SCALER_Y_JOB,
            "parity_png": PARITY_PNG, "residuals_png": RESID_PNG, "f9_png": F9_PNG,
            "trials_xlsx": TRIALS_XLSX, "trials_png": TRIALS_PNG, "optuna_db": OPTUNA_DB
        }
    }, f, indent=2, ensure_ascii=False)
print("Done:", RESULTS_JSON)
