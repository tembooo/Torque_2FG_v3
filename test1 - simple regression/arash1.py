# Arash1.py
import os
import sys
import gc
import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Run without GUI
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf

import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ==============================
# Project paths (Windows)
# ==============================
BASE_DIR = r"C:\12.LUT\00.Termic Cources\0.Finished\Arash\4.4th project\M1.Torque_2FG_v3"
if not os.path.isdir(BASE_DIR):
    raise FileNotFoundError(f"BASE_DIR not found:\n{BASE_DIR}")

FILE_TORQUE = os.path.join(BASE_DIR, "1.Torque_2FG_v3.xlsx")

TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_PATH             = os.path.join(BASE_DIR, f"run_log_{TS}.txt")
SCALER_PATH          = os.path.join(BASE_DIR, "scaler_torque.joblib")
OPTUNA_DB_FILE       = os.path.join(BASE_DIR, "torque_optuna.db")
OPTUNA_DB_URL        = f"sqlite:///{OPTUNA_DB_FILE.replace(os.sep, '/')}"
OPTUNA_HISTORY_PNG   = os.path.join(BASE_DIR, "optuna_history.png")
TRAINING_HISTORY_CSV = os.path.join(BASE_DIR, "training_history.csv")
TRAINING_LOSS_PNG    = os.path.join(BASE_DIR, "training_loss.png")
TOP5_TRIALS_TXT      = os.path.join(BASE_DIR, "top5_trials.txt")
MODEL_PATH           = os.path.join(BASE_DIR, "final_mlp_torque.h5")
BEST_PARAMS_TXT      = os.path.join(BASE_DIR, "best_params.txt")
PREDICTIONS_XLSX     = os.path.join(BASE_DIR, "Torque_predictions.xlsx")

EDA_CORR_PNG    = os.path.join(BASE_DIR, "eda_corr.png")
EDA_BOXPLOT_PNG = os.path.join(BASE_DIR, "eda_boxplot.png")
EDA_LAG_PNG     = os.path.join(BASE_DIR, "eda_lag.png")
EDA_ACF_PNG     = os.path.join(BASE_DIR, "eda_acf.png")

# ==============================
# Config
# ==============================
BATCH_SIZE = 32  # ثابت نگه می‌داریم تا KeyError پیش نیاید
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU growth (اختیاری)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print("TF memory growth enabled.")
    except Exception as e:
        print(f"Could not set memory growth: {e}")

# Logging (اختیاری: ریدایرکت)
log_file = open(LOG_PATH, "w", buffering=1, encoding="utf-8")
# sys.stdout = log_file
# sys.stderr = log_file

print(f"BASE_DIR: {BASE_DIR}")
print(f"Start: {TS}")

# ==============================
# Load data
# ==============================
if not os.path.exists(FILE_TORQUE):
    raise FileNotFoundError(f"Excel not found:\n{FILE_TORQUE}")

df_raw = pd.read_excel(FILE_TORQUE)
print("Columns:", df_raw.columns.tolist())

input_columns = [f"F{i}" for i in range(1, 10)]
target_column = "Torque"
missing = [c for c in input_columns + [target_column] if c not in df_raw.columns]
if missing:
    raise ValueError(f"Missing columns in Excel: {missing}")

df = df_raw[input_columns + [target_column]].copy()

# ==============================
# EDA -> files
# ==============================
def explore_and_save(df, target_name):
    df_num = df.select_dtypes(include=[np.number])

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_num.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(EDA_CORR_PNG, dpi=150); plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_num, orient="h")
    plt.title("Boxplot")
    plt.tight_layout()
    plt.savefig(EDA_BOXPLOT_PNG, dpi=150); plt.close()

    plt.figure(figsize=(6, 6))
    lag_plot(df_num[target_name])
    plt.title("Lag Plot")
    plt.tight_layout()
    plt.savefig(EDA_LAG_PNG, dpi=150); plt.close()

    plt.figure(figsize=(10, 5))
    plot_acf(df_num[target_name], lags=30)
    plt.title("Autocorrelation (ACF)")
    plt.tight_layout()
    plt.savefig(EDA_ACF_PNG, dpi=150); plt.close()

    print("EDA plots saved.")

explore_and_save(df, target_column)

# ==============================
# Scale
# ==============================
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[input_columns] = scaler.fit_transform(df_scaled[input_columns])
joblib.dump(scaler, SCALER_PATH)
print(f"Scaler saved: {SCALER_PATH}")

# ==============================
# Split
# ==============================
train_df, test_df = train_test_split(df_scaled, test_size=0.2, random_state=42, shuffle=True)

def prepare_data(tr_df, te_df):
    X_tr = tr_df[input_columns].values
    y_tr = tr_df[target_column].values
    X_te = te_df[input_columns].values
    y_te = te_df[target_column].values
    return X_tr, y_tr, X_te, y_te

# ==============================
# Model
# ==============================
def build_model(hp, input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),            # حذف هشدار input_dim
        Dense(hp['units1'], activation='relu'),
        Dense(hp['units2'], activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=hp['learning_rate']),
                  loss='mse', metrics=['mae'])
    return model

# ==============================
# Optuna (fallback-safe)
# ==============================
best_hp = None
optuna_available = True
try:
    import optuna
    from optuna.visualization.matplotlib import plot_optimization_history

    def objective(trial):
        hp = {
            'units1': trial.suggest_int('units1', 32, 256, step=32),
            'units2': trial.suggest_int('units2', 16, 128, step=16),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            # batch_size را در study ذخیره نمی‌کنیم تا با مطالعه‌های قبلی سازگار بماند
        }
        X_tr, y_tr, X_te, y_te = prepare_data(train_df, test_df)
        model = build_model(hp, X_tr.shape[1])

        cbs = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6),
        ]

        model.fit(X_tr, y_tr, validation_split=0.1, epochs=200,
                  batch_size=BATCH_SIZE, callbacks=cbs, verbose=0)
        y_pred = model.predict(X_te, verbose=0)
        mse = mean_squared_error(y_te, y_pred)
        tf.keras.backend.clear_session(); gc.collect()
        return mse

    storage = optuna.storages.RDBStorage(url=OPTUNA_DB_URL)
    study = optuna.create_study(direction='minimize',
                                study_name="mlp_torque_tuning",
                                storage=storage, load_if_exists=True)

    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        study.optimize(objective, n_trials=20)

    try:
        fig = plot_optimization_history(study)
        fig.figure.tight_layout()
        fig.figure.savefig(OPTUNA_HISTORY_PNG)
        plt.close()
        print(f"Optuna history saved: {OPTUNA_HISTORY_PNG}")
    except Exception as _:
        pass

    best_hp = study.best_params if study.best_trial else None
    print("Best hyperparameters:", best_hp)

except Exception as e:
    optuna_available = False
    print(f"Optuna unavailable or failed ({e}). Using fallback hyperparams.")

if best_hp is None:
    best_hp = {'units1': 128, 'units2': 64, 'learning_rate': 1e-3}

# ==============================
# Final training
# ==============================
X_train, y_train, X_test, y_test = prepare_data(train_df, test_df)
final_model = build_model(best_hp, X_train.shape[1])

history = final_model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=200,
    batch_size=BATCH_SIZE,   # ثابت
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    ],
    verbose=2
)

pd.DataFrame(history.history).to_csv(TRAINING_HISTORY_CSV, index=False)

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(TRAINING_LOSS_PNG, dpi=150); plt.close()
print(f"Training loss saved: {TRAINING_LOSS_PNG}")

# ==============================
# Evaluation
# ==============================
y_pred = final_model.predict(X_test, verbose=0).flatten()
print(f"Test MSE: {mean_squared_error(y_test, y_pred):.6f}")
print(f"Test MAE: {mean_absolute_error(y_test, y_pred):.6f}")
print(f"Test R^2: {r2_score(y_test, y_pred):.6f}")

# ==============================
# Save model/results
# ==============================
final_model.save(MODEL_PATH)
with open(BEST_PARAMS_TXT, "w", encoding="utf-8") as f:
    f.write(str({'batch_size': BATCH_SIZE, **best_hp}))  # برای شفافیت، batch_size هم نوشته می‌شود

scaler_loaded = joblib.load(SCALER_PATH)
X_all_scaled = scaler_loaded.transform(df[input_columns])
df_out = df_raw.copy()
df_out["Torque_pred"] = final_model.predict(X_all_scaled, verbose=0).flatten()
df_out.to_excel(PREDICTIONS_XLSX, index=False)

if optuna_available:
    try:
        top_trials = sorted(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
            key=lambda t: t.value
        )[:5]
        with open(TOP5_TRIALS_TXT, "w", encoding="utf-8") as f:
            for i, t in enumerate(top_trials, 1):
                f.write(f"Trial {i} - MSE: {t.value:.6f} - Params: {t.params}\n")
    except Exception:
        pass

print("All outputs saved to:", BASE_DIR)
log_file.close()
