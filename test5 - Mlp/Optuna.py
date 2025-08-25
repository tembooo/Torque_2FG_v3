import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_error
import joblib
import optuna

# ------------------------------
# Paths
# ------------------------------
BASE_DIR = r"C:\12.LUT\00.Termic Cources\0.Finished\Arash\4.4th project\M1.Torque_2FG_v3"
FILE_XLSX = os.path.join(BASE_DIR, "1.Torque_2FG_v3.xlsx")

PRED_XLSX   = os.path.join(BASE_DIR, "predictions_mlp_optuna.xlsx")
RESULTS_JSON = os.path.join(BASE_DIR, "mlp_optuna_results.json")
BEST_MODEL_H5 = os.path.join(BASE_DIR, "best_mlp_model.h5")
PARITY_PNG = os.path.join(BASE_DIR, "parity_mlp.png")
RESID_PNG  = os.path.join(BASE_DIR, "residuals_mlp.png")
F9_PNG     = os.path.join(BASE_DIR, "actual_vs_pred_F9_mlp.png")

# ------------------------------
# Reproducibility
# ------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------
# Load data
# ------------------------------
df = pd.read_excel(FILE_XLSX)
input_cols = [f"F{i}" for i in range(1, 10)]
target_col = "Torque"

X = df[input_cols].copy().values
y = df[target_col].values.reshape(-1, 1)

# Scale inputs and outputs
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# Split
X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.2, random_state=SEED, shuffle=True)

# ------------------------------
# Build model function
# ------------------------------
def build_mlp(n_layers, n_units, dropout_rate, input_dim, lr):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for _ in range(n_layers):
        model.add(layers.Dense(n_units, activation="relu"))
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1))  # regression output
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="mse")
    return model

# ------------------------------
# Optuna objective
# ------------------------------
def objective(trial):
    n_layers = trial.suggest_int("n_layers", 1, 4)
    n_units = trial.suggest_int("n_units", 32, 256, step=32)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.4)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = 200

    model = build_mlp(n_layers, n_units, dropout_rate, X_train.shape[1], lr)

    es = callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[es]
    )
    val_loss = min(history.history["val_loss"])
    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=40, show_progress_bar=False)

best_params = study.best_trial.params
print("Best params:", best_params)

# ------------------------------
# Train final model
# ------------------------------
final_model = build_mlp(
    best_params["n_layers"],
    best_params["n_units"],
    best_params["dropout_rate"],
    X_train.shape[1],
    best_params["lr"]
)

es = callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
final_model.fit(
    X_train, y_train,
    validation_data=(X_hold, y_hold),
    epochs=500,
    batch_size=best_params["batch_size"],
    callbacks=[es],
    verbose=1
)

final_model.save(BEST_MODEL_H5)
print("✅ Best MLP model saved:", BEST_MODEL_H5)

# ------------------------------
# Evaluate
# ------------------------------
y_pred_scaled = final_model.predict(X_hold)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_hold)

mse = mean_squared_error(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2  = r2_score(y_true, y_pred)
metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
print("Holdout metrics (MLP):", metrics)

# ------------------------------
# Save predictions
# ------------------------------
y_all_pred_scaled = final_model.predict(X)
y_all_pred = scaler_y.inverse_transform(y_all_pred_scaled)
out = pd.DataFrame(df.copy())
out["Torque_pred_MLP"] = y_all_pred
out.to_excel(PRED_XLSX, index=False)
print("✅ Predictions saved:", PRED_XLSX)

# ------------------------------
# Plots
# ------------------------------
# Parity
plt.figure(figsize=(6,6))
mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
plt.scatter(y_true, y_pred, alpha=0.6)
plt.plot([mn, mx], [mn, mx], "--")
plt.xlabel("Actual Torque")
plt.ylabel("Predicted Torque")
plt.title("Parity Plot (MLP)")
plt.grid(True); plt.tight_layout()
plt.savefig(PARITY_PNG, dpi=150); plt.close()

# Residuals
resid = y_pred.flatten() - y_true.flatten()
plt.figure(figsize=(10,4))
plt.plot(resid, marker="o", linestyle="None", alpha=0.6)
plt.axhline(0, color="k", linestyle="--")
plt.title("Residuals (Holdout) | MLP")
plt.xlabel("Sample index")
plt.ylabel("Residual")
plt.grid(True); plt.tight_layout()
plt.savefig(RESID_PNG, dpi=150); plt.close()

# Actual vs Pred by F9
plt.figure(figsize=(10,6))
plt.scatter(df["F9"], df["Torque"], alpha=0.6, label="Actual")
plt.scatter(df["F9"], out["Torque_pred_MLP"], alpha=0.6, label="Predicted")
plt.xlabel("F9")
plt.ylabel("Torque")
plt.title("Actual vs Predicted Torque by F9 (MLP)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(F9_PNG, dpi=150); plt.close()

# ------------------------------
# Save results json
# ------------------------------
results = {
    "best_params": best_params,
    "metrics": metrics,
    "paths": {
        "predictions": PRED_XLSX,
        "model": BEST_MODEL_H5,
        "parity_png": PARITY_PNG,
        "residuals_png": RESID_PNG,
        "f9_png": F9_PNG
    }
}
with open(RESULTS_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Done. Results saved to:", RESULTS_JSON)
