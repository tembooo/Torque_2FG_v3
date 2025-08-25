# app_mlp.py
import os
import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import joblib

# ------------------------------
# Paths
# ------------------------------
MODEL_PATH = "best_mlp_model.keras"
SCALER_X_PATH = "scaler_X.joblib"
SCALER_Y_PATH = "scaler_y.joblib"

FEATURES = [f"F{i}" for i in range(1, 10)]

# ------------------------------
# Load model and scalers
# ------------------------------
@st.cache_resource
def load_model_and_scalers():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    if not os.path.exists(SCALER_X_PATH) or not os.path.exists(SCALER_Y_PATH):
        st.error("Scaler files not found. Please save scalers during training with joblib.")
        st.stop()

    model = keras.models.load_model(MODEL_PATH)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_model_and_scalers()

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Torque Prediction (MLP Best)")
st.write("Enter values for F1..F9 to predict Torque.")

# ---- 1) Single Prediction ----
st.subheader("Single Prediction")

vals = {}
cols = st.columns(3)
for i, f in enumerate(FEATURES):
    with cols[i % 3]:
        vals[f] = st.number_input(f, value=0.0, format="%.6f")

if st.button("Predict Torque"):
    X = pd.DataFrame([vals], columns=FEATURES)
    X_scaled = scaler_X.transform(X.values)
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0, 0]
    st.success(f"Predicted Torque: {y_pred:.6f}")

st.divider()

# ---- 2) Batch Prediction ----
st.subheader("Batch Prediction (Excel/CSV)")
up = st.file_uploader("Upload file with columns F1..F9", type=["xlsx", "xls", "csv"])
if up is not None:
    if up.name.lower().endswith((".xlsx", ".xls")):
        df_in = pd.read_excel(up)
    else:
        df_in = pd.read_csv(up)

    miss = [c for c in FEATURES if c not in df_in.columns]
    if miss:
        st.error(f"Missing columns: {miss}")
    else:
        X_scaled = scaler_X.transform(df_in[FEATURES].values)
        y_pred_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
        df_out = df_in.copy()
        df_out["Torque_pred"] = y_pred
        st.dataframe(df_out.head(20))
        st.download_button(
            "Download predictions",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name="predictions_mlp.csv",
            mime="text/csv"
        )
