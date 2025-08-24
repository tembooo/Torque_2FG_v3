# Part Five MLP instead of the simple regression 
 Torque Prediction with MLP + Optuna
**Project type:** Supervised regression (predicting `Torque` from features `F1`–`F9`)  
**Frameworks:** TensorFlow/Keras, Optuna, scikit‑learn, Matplotlib  
**Author/runtime:** Script-driven, Windows paths supported

---

 1) Executive Summary
This project builds a **Multi-Layer Perceptron (MLP)** regressor to predict **Torque** using nine input features (`F1`–`F9`).  
The pipeline standardizes inputs/targets, splits data into train/holdout sets, tunes the MLP hyperparameters with **Optuna** (40 trials, minimizing validation MSE), retrains a final model on the training set with **EarlyStopping**, and evaluates on a **holdout** split.  
Deliverables include the trained model (`.h5`), predictions (`.xlsx`), plots (parity, residuals, and F9 vs Torque), and a `results.json` manifest.

---

 2) Data & Paths
- **Input file:** `1.Torque_2FG_v3.xlsx` (sheet with columns `F1..F9` and `Torque`).
- **Target:** `Torque` (continuous).
- **Feature matrix:** `X = [F1, F2, ..., F9]`  
- **Output artifacts (auto‑generated):**
  - `best_mlp_model.h5` – Trained Keras model.
  - `predictions_mlp_optuna.xlsx` – Original data + `Torque_pred_MLP`.
  - `parity_mlp.png` – Actual vs Predicted scatter with 45° line.
  - `residuals_mlp.png` – Residuals on holdout set.
  - `actual_vs_pred_F9_mlp.png` – Scatter by `F9` (Actual vs Predicted).
  - `mlp_optuna_results.json` – Best hyperparameters, metrics, and artifact paths.

> **Note:** Paths are defined at the top of the script. Adjust `BASE_DIR` to your project folder.

---

 3) Reproducibility
- Global seeds are set for `random`, `numpy`, and `tensorflow` (`SEED = 42`).
- Dropout layers introduce RNG; with fixed seeds and a single device the run should be largely reproducible.
- `matplotlib` runs in headless mode via `Agg` backend, so plots are always saved to disk.

---

 4) Preprocessing
1. Read Excel into a pandas DataFrame.
2. Select inputs `F1..F9` and target `Torque`.
3. **Standardize** both `X` and `y` with `StandardScaler()` (zero mean, unit variance).
4. **Train/Holdout split**: 80/20 with `train_test_split(..., shuffle=True, random_state=SEED)`.

> Scaling the **target** (`y`) helps stable training for neural nets. For reporting metrics, predictions are **inverse‑transformed** back to the original torque scale.

---

 5) Model Architecture (Keras MLP)
```text
Input: 9 features
[Repeat n_layers times]:
  Dense(n_units, activation="relu")
  Dropout(dropout_rate)
Output: Dense(1)  # scalar torque
Loss: Mean Squared Error (MSE)
Optimizer: Adam(lr)
```
- The number of layers, units, dropout rate, and learning rate are **searchable** via Optuna.
- The output layer is linear (default) for regression.

---

 6) Hyperparameter Optimization (Optuna)
**Objective:** minimize validation loss (MSE) on a **20% validation split of the training set**.  
**Search space:**

| Parameter     | Type / Range                            |
|---------------|------------------------------------------|
| `n_layers`    | integer ∈ [1, 4]                         |
| `n_units`     | integer ∈ {32, 64, 96, …, 256} (step=32) |
| `dropout_rate`| float ∈ [0.0, 0.4]                       |
| `lr`          | float log‑uniform ∈ [1e‑4, 1e‑2]         |
| `batch_size`  | categorical ∈ {16, 32, 64}               |
| Epochs/Trial  | 200 with `EarlyStopping(patience=20)`    |

> The objective returns `min(history.history["val_loss"])`. Optuna runs **40 trials** and records the best trial.

---

 7) Final Training
After tuning, the script rebuilds the MLP with the **best parameters** and trains it with:
- `validation_data=(X_hold, y_hold)`
- `epochs=500`
- `EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)`

The trained model is saved to `best_mlp_model.h5`.

> **Note:** The final fit uses the **holdout as validation** (not for weight updates after early stopping). You may later retrain on `train+holdout` once hyperparameters are fixed, but keep a **separate test set** if you do so.

---

 8) Evaluation & Metrics
Metrics are computed on the **holdout set** after inverse transforming predictions:
- **MAE**: \( \frac{1}{n}\sum |y_i - \hat{y}_i| \)
- **MSE**: \( \frac{1}{n}\sum (y_i - \hat{y}_i)^2 \)
- **RMSE**: \( \sqrt{\text{MSE}} \)
- **R²**: \( 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} \)

The script also produces:
- **Parity plot** (`parity_mlp.png`) — ideal predictions lie on the diagonal.
- **Residual plot** (`residuals_mlp.png`) — checks homoscedasticity & bias (residuals clustering around 0).
- **Actual vs Predicted by F9** (`actual_vs_pred_F9_mlp.png`) — sanity‑check for one influential feature.

> The exact metric values depend on your dataset and random seed; the script prints a dictionary like:
```python
{"MSE": <float>, "RMSE": <float>, "MAE": <float>, "R2": <float>}
```

---

 9) Saved Predictions
The script predicts on **all rows** and exports:
- `predictions_mlp_optuna.xlsx`  
  Columns: original dataset + `Torque_pred_MLP` in the **original torque scale**.

---

 10) Results Manifest (`mlp_optuna_results.json`)
Example structure (values shown as placeholders):
```json
{
  "best_params": {
    "n_layers": 3,
    "n_units": 128,
    "dropout_rate": 0.15,
    "lr": 0.0008,
    "batch_size": 32
  },
  "metrics": {
    "MSE": 12.34,
    "RMSE": 3.51,
    "MAE": 2.10,
    "R2": 0.9981
  },
  "paths": {
    "predictions": "...\predictions_mlp_optuna.xlsx",
    "model": "...\best_mlp_model.h5",
    "parity_png": "...\parity_mlp.png",
    "residuals_png": "...\residuals_mlp.png",
    "f9_png": "...\actual_vs_pred_F9_mlp.png"
  }
}
```
Use this file to programmatically read best hyperparameters and artifact locations.

---

11) How to Run
1. **Install dependencies**
   ```bash
   pip install numpy pandas scikit-learn matplotlib tensorflow optuna joblib openpyxl
   ```
2. **Adjust paths** at the top of the script (set `BASE_DIR` and ensure `1.Torque_2FG_v3.xlsx` exists).
3. **Run the script**
   ```bash
   python your_script_name.py
   ```
4. **Collect artifacts** from the defined output paths.

---

12) Design Choices & Rationale
- **Standardization for X and y** supports stable gradient updates and balances feature scales.
- **EarlyStopping** avoids overfitting and shortens unnecessary epochs; `restore_best_weights=True` returns the best checkpoint automatically.
- **Optuna** efficiently explores the space of depth/width/regularization/LR/batch size.
- **Holdout evaluation** provides an unbiased check separate from the validation split used during tuning.

---

13) Limitations & Next Steps
- **K‑Fold CV** for Optuna: replace `validation_split` with cross‑validated scoring to reduce variance from one split.
- **Learning‑rate scheduling**: add `ReduceLROnPlateau` or cosine decay for better convergence.
- **Regularization**: consider `L2` on Dense layers in addition to dropout.
- **Feature engineering**: test interactions or non‑linear transforms for better expressiveness.
- **Model baselines**: compare to tree ensembles (e.g., HistGBR/XGBoost) to validate MLP benefits.
- **Pipeline packaging**: persist `scaler_X` and `scaler_y` with `joblib.dump()` for later inference scripts.

---

14) Code Pointers (where to modify)
- **Search space** → inside `objective(trial)` (`n_layers`, `n_units`, `dropout_rate`, `lr`, `batch_size`).
- **Network depth/width** → `build_mlp(...)` loop.
- **Loss/metrics** → `model.compile(...)` (currently `"mse"` loss).
- **Plots** → final section; extend with feature‑wise diagnostics or error histograms.
- **Artifacts** → change file names at the top in the *Paths* block.

---

15) References
- **Optuna**: Efficient hyperparameter optimization framework.
- **Keras**: High‑level TensorFlow API for building neural networks.
- **scikit‑learn**: Preprocessing, metrics, and train/test utilities.
- **Matplotlib**: Static plotting backend (`Agg`) for saved figures.

---








# Part Four Continuous Regression  and baseline is HistGBR
there is difference between "Continuous Regression" and "Stepwise Regression with Trees" so for this project we need to use 

| Model             | MSE     | RMSE  | MAE   | R²      |
| ----------------- | ------- | ----- | ----- | ------- |
| **HistGBR\_raw**  | 41.21   | 6.42  | 2.61  | 0.99973 |
| **HistGBR\_rule** | 38.62   | 6.21  | 2.29  | 0.99974 |
| **Linear**        | 2865.72 | 53.53 | 37.31 | 0.98092 |
| **Polynomial**    | 968.98  | 31.13 | 18.84 | 0.99355 |

- **HistGBR (Histogram-based Gradient Boosting Regressor)**  
  - Achieves the **best performance**, with extremely high R² (~0.9997).  
  - Very low errors (MAE ≈ 2.3–2.6, RMSE ≈ 6.2).  
  - Captures the nonlinear relationship between F9 and Torque very effectively.  

- **Linear Regression**  
  - R² = 0.98, but high errors (MAE ≈ 37, RMSE ≈ 53).  
  - Too simplistic, fails to model the nonlinear dynamics.  

- **Polynomial Regression**  
  - R² = 0.9936, but still higher errors (MAE ≈ 18.8, RMSE ≈ 31.1) compared to boosting.  
  - Deviates from actual data at certain ranges.  


Gradient boosting (HistGBR) provides the **most accurate and reliable torque predictions**.  
Linear regression is insufficient, and polynomial regression only partially improves performance but still falls short compared to boosting.
<p align="center">
  <img src="https://github.com/tembooo/Torque_2FG_v3/blob/main/pic3.Polynomial.png" width="700" alt="Torque vs Features">
</p>


but the performance of the HistGBR still is better and we used the optuna as well but result is not good. 

so based on this result I decided to establish new neural network (Mlp) I hope that would be better result. 

# Part 3.2 optuna to find the best hyper parameters. 

Best Hyperparameters (Optuna)

Number of trials: 40

Best Trial (28):

| Hyperparameter      | Value   |
| ------------------- | ------- |
| learning\_rate      | 0.1056  |
| max\_iter           | 700     |
| max\_depth          | 8       |
| min\_samples\_leaf  | 9       |
| l2\_regularization  | 0.02594 |
| max\_bins           | 222     |
| n\_iter\_no\_change | 50      |

| Setting              | MSE       | RMSE     | MAE      | R²           |
| -------------------- | --------- | -------- | -------- | ------------ |
| **Best CV (Optuna)** | –         | **8.36** | –        | –            |
| Holdout (Raw)        | 45.93     | 6.78     | 2.59     | 0.999694     |
| Holdout (With Rule)  | **44.39** | **6.66** | **2.47** | **0.999704** |

# Part 3.1 find the best model

| Model          | MSE       | RMSE     | MAE      | R²          | CV R² mean | CV R² std | CV MAE mean | CV RMSE mean |
| -------------- | --------- | -------- | -------- | ----------- | ---------- | --------- | ----------- | ------------ |
| RandomForest   | 115.60    | 10.75    | 3.38     | 0.99923     | 0.99903    | 0.00023   | 3.69        | 11.94        |
| HistGBR (Best) | **46.51** | **6.82** | 2.92     | **0.99969** | 0.99940    | 0.00021   | 3.44        | 9.38         |
| MLP            | 58.86     | 7.67     | **2.86** | 0.99961     | –          | –         | –           | –            |



# Part Two: Investigating the Features

## 📈 Results & Analysis

Below is a sample visualization of torque vs. input features.  
*(Note: “F9” refers to the **Current** feature in the plot.)*

<p align="center">
  <img src="https://github.com/tembooo/Torque_2FG_v3/blob/main/pic1.Torque_vs_Features.png" width="700" alt="Torque vs Features">
</p>

**Observation:** The plot indicates that **current** shows a strong relationship with **torque**, and it also appears correlated with several other features.  
We will quantify these relationships next (e.g., correlation coefficients, permutation importance, and model-based feature importance).


and we can see the most important feateures for this modeling is "current". 
<p align="center">
  <img src="https://github.com/tembooo/Torque_2FG_v3/blob/main/pic2.feature_importance.png" width="700" alt="Torque vs Features">
</p>

---

# Paert one: Torque_2FG_v3

This repository contains data and resources for building a predictive model of **torque** based on electrical machine parameters.  
The project is based on datasets provided by **Arash** and aims to develop and evaluate machine learning models for accurate torque estimation.

---

## 📂 Repository Structure

- **1.Torque_2FG_v3.xlsx**  
  Dataset containing torque measurement data (target variable).
  [Download here](https://github.com/tembooo/Torque_2FG_v3/blob/main/1.Torque_2FG_v3.xlsx)

- **2.Ld-Lq_2FG_v3.xlsx**  
  Dataset containing `Ld` and `Lq` parameters (input features for the model).
  [Download here](https://github.com/tembooo/Torque_2FG_v3/blob/main/2.Ld-Lq_2FG_v3.xlsx)
  
---

## 🎯 Project Objective

The goal of this project is to **predict torque** in a two-phase generator (2FG) system using machine learning models trained on the provided datasets.  
By leveraging the relationship between `Ld`, `Lq`, and torque, we aim to create a robust model that can be used for analysis, optimization, and control purposes.
