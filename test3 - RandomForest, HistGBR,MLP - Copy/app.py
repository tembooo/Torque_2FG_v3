# app.py
import os
import pandas as pd
import joblib
import streamlit as st
from sklearn.base import BaseEstimator, RegressorMixin, clone

FEATURES = [f"F{i}" for i in range(1, 10)]
MODEL_PATH = "best_histgbr_optuna.joblib"
RAW_MODEL_PATH = "best_histgbr_optuna.joblib"  # fallback

# ---- 1) تعریف کلاس مثل زمان آموزش (قبل از joblib.load) ----
class ZeroCurrentZeroTorqueWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator=None, current_feature="F9", zero_current_weight=5.0):
        self.base_estimator = base_estimator
        self.current_feature = current_feature
        self.zero_current_weight = zero_current_weight

    def fit(self, X, y):
        self.feature_names_in_ = list(X.columns) if hasattr(X, "columns") else [f"F{i}" for i in range(1, X.shape[1]+1)]
        self.current_idx_ = self.feature_names_in_.index(self.current_feature)
        x_np = X.values if hasattr(X, "values") else X
        import numpy as np
        w = np.ones(len(y), dtype=float)
        w[x_np[:, self.current_idx_] == 0] = self.zero_current_weight
        self.model_ = clone(self.base_estimator)
        self.model_.fit(X, y, sample_weight=w)
        return self

    def predict(self, X):
        x_np = X.values if hasattr(X, "values") else X
        import numpy as np
        preds = self.model_.predict(X)
        preds = np.asarray(preds, dtype=float)
        mask_zero = (x_np[:, self.current_idx_] == 0)
        preds[mask_zero] = 0.0
        return preds

# ---- 2) لود مدل با کش ----
@st.cache_resource
def load_model():
    # اول تلاش برای مدل wrap شده
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH), "with_rule"
        except Exception as e:
            st.warning(f"Could not load {MODEL_PATH}: {e}")

    # اگر نشد، مدل خام را لود کن و قانون را در زمان پیش‌بینی اعمال کن
    if os.path.exists(RAW_MODEL_PATH):
        try:
            raw = joblib.load(RAW_MODEL_PATH)
            return raw, "raw"
        except Exception as e:
            st.error(f"Could not load fallback {RAW_MODEL_PATH}: {e}")
            st.stop()

    st.error("No model file found.")
    st.stop()

st.title("Torque Prediction (HistGBR + Optuna)")
st.write("Enter F1..F9 and get predicted Torque. If current (F9) = 0, torque is forced to 0 by rule.")

model, mode = load_model()
st.caption(f"Loaded model mode: **{mode}**")

# ---- 3) فرم پیش‌بینی تکی ----
st.subheader("Single Prediction")
vals = {}
cols = st.columns(3)
for i, f in enumerate(FEATURES):
    with cols[i % 3]:
        vals[f] = st.number_input(f, value=0.0, format="%.6f")

if st.button("Predict Torque"):
    X = pd.DataFrame([vals], columns=FEATURES)
    y = model.predict(X)[0]
    # اگر مدل خام هست، قاعده را اینجا هم اعمال کن
    if mode == "raw" and float(X["F9"].iloc[0]) == 0.0:
        y = 0.0
    st.success(f"Predicted Torque: {float(y):.6f}")

st.divider()

# ---- 4) پیش‌بینی دسته‌ای با فایل ----
st.subheader("Batch Prediction (Excel/CSV)")
up = st.file_uploader("Upload a file with columns F1..F9", type=["xlsx", "xls", "csv"])
if up is not None:
    if up.name.lower().endswith((".xlsx", ".xls")):
        df_in = pd.read_excel(up)
    else:
        df_in = pd.read_csv(up)
    miss = [c for c in FEATURES if c not in df_in.columns]
    if miss:
        st.error(f"Missing columns: {miss}")
    else:
        preds = model.predict(df_in[FEATURES])
        # اگر مدل خام است، قانون را پس‌پردازش کن
        if mode == "raw":
            import numpy as np
            preds = np.asarray(preds, dtype=float)
            preds[df_in["F9"].values == 0] = 0.0
        df_out = df_in.copy()
        df_out["Torque_pred"] = preds
        st.dataframe(df_out.head(20))
        st.download_button(
            "Download predictions",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv"
        )
