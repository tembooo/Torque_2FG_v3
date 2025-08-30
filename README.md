# Part Five MLP for LD - LQ 

**Data target:** `Ld-Lq`  
**Features:** `F1..F10`

<p align="center">
  <img src="https://github.com/tembooo/Torque_2FG_v3/blob/main/pic5-best optuna ld-lq.png" width="700" alt="Torque vs Features">
</p>
Best Hyperparameters

| n_layers | n_units | dropout_rate | lr | batch_size |
|---:|---:|---:|---:|---:|
| 2 | 224 | 0.031500 | 0.002312572 | 32 |

Top Trials (by lowest val_loss)

| rank | trial | val_loss | n_layers | n_units | dropout_rate | lr | batch_size |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 0.0001627001 | 2 | 224 | 0.031500 | 0.002312572 | 32 |
| 2 | 33 | 0.0001870142 | 2 | 224 | 0.029633 | 0.002417494 | 32 |
| 3 | 34 | 0.0002010334 | 2 | 224 | 0.025843 | 0.002156666 | 64 |
| 4 | 21 | 0.0002179104 | 2 | 160 | 0.045276 | 0.001823978 | 32 |
| 5 | 39 | 0.0002208960 | 2 | 224 | 0.072628 | 0.001932214 | 16 |
| 6 | 31 | 0.0002832184 | 2 | 224 | 0.032128 | 0.002261640 | 32 |
| 7 | 18 | 0.0003518902 | 2 | 96 | 0.047105 | 0.003041606 | 32 |
| 8 | 15 | 0.0003970739 | 2 | 160 | 0.054439 | 0.001369654 | 32 |


All hyper parameters 
| trial |   val\_loss | n\_layers | n\_units | dropout\_rate |          lr | batch\_size |
| ----: | ----------: | --------: | -------: | ------------: | ----------: | ----------: |
|     0 | 0.006258796 |         1 |       32 |      0.250328 | 0.000301087 |          32 |
|     1 | 0.015711579 |         3 |      256 |      0.278794 | 0.000266954 |          64 |
|     2 | 0.034696691 |         4 |      128 |      0.323178 | 0.000280739 |          64 |
|     3 | 0.001554713 |         1 |       96 |      0.107951 | 0.001592105 |          16 |
|     4 | 0.001069017 |         1 |      224 |      0.255068 | 0.000447758 |          16 |
|     5 | 0.000685145 |         2 |      256 |      0.096064 | 0.008526722 |          16 |
|     6 | 0.004543237 |         3 |       32 |      0.367428 | 0.005849415 |          16 |
|     7 | 0.011698360 |         4 |      160 |      0.168289 | 0.000201244 |          64 |
|     8 | 0.061291683 |         4 |       96 |      0.369677 | 0.000422948 |          64 |
|     9 | 0.000630306 |         1 |      256 |      0.064163 | 0.000203560 |          16 |
|    10 | 0.000581804 |         2 |      192 |      0.016106 | 0.000104796 |          32 |
|    11 | 0.000475103 |         2 |      192 |      0.004631 | 0.000103211 |          32 |
|    12 | 0.000493063 |         2 |      192 |      0.005049 | 0.000111473 |          32 |
|    13 | 0.000467770 |         2 |      192 |      0.001047 | 0.000112213 |          32 |
|    14 | 0.004787291 |         3 |      192 |      0.164494 | 0.000989297 |          32 |
|    15 | 0.000397074 |         2 |      160 |      0.054439 | 0.001369654 |          32 |
|    16 | 0.000585186 |         2 |      128 |      0.073589 | 0.002186267 |          32 |
|    17 | 0.008379768 |         3 |      160 |      0.131065 | 0.000842491 |          32 |
|    18 | 0.000351890 |         2 |       96 |      0.047105 | 0.003041606 |          32 |
|    19 | 0.001130078 |         3 |       64 |      0.055372 | 0.003238463 |          32 |
|    20 | 0.001683096 |         1 |       96 |      0.194873 | 0.003711888 |          32 |
|    21 | 0.000217910 |         2 |      160 |      0.045276 | 0.001823978 |          32 |
|    22 | 0.000503308 |         2 |      128 |      0.045864 | 0.001663760 |          32 |
|    23 | 0.000506712 |         2 |      160 |      0.126125 | 0.002879012 |          32 |
|    24 | 0.001583282 |         2 |       64 |      0.087237 | 0.000720067 |          32 |
|    25 | 0.000787239 |         3 |      128 |      0.039057 | 0.001471234 |          32 |
|    26 | 0.003380428 |         2 |       64 |      0.150428 | 0.004287273 |          32 |
|    27 | 0.000727409 |         1 |      160 |      0.035985 | 0.002127740 |          32 |
|    28 | 0.001280488 |         2 |      224 |      0.105689 | 0.000590794 |          64 |
|    29 | 0.001622260 |         1 |       96 |      0.207259 | 0.001160750 |          32 |
|    30 | 0.011393762 |         3 |       32 |      0.217568 | 0.005288762 |          32 |
|    31 | 0.000283218 |         2 |      224 |      0.032128 | 0.002261640 |          32 |
|    32 | 0.000162700 |         2 |      224 |      0.031500 | 0.002312572 |          32 |
|    33 | 0.000187014 |         2 |      224 |      0.029633 | 0.002417494 |          32 |
|    34 | 0.000201033 |         2 |      224 |      0.025843 | 0.002156666 |          64 |
|    35 | 0.000644291 |         1 |      224 |      0.079567 | 0.002364403 |          64 |
|    36 | 0.000863215 |         2 |      256 |      0.317159 | 0.007533509 |          64 |
|    37 | 0.000236150 |         3 |      224 |      0.024132 | 0.004723887 |          64 |
|    38 | 0.000822656 |         1 |      256 |      0.112951 | 0.001619746 |          64 |
|    39 | 0.000220896 |         2 |      224 |      0.072628 | 0.001932214 |          16 |




# Part Five MLP instead of the simple regression 
the result with the mlp: 
| Metric   | Value   |
| -------- | ------- |
| **MSE**  | 42.21   |
| **RMSE** | 6.50    |
| **MAE**  | 2.84    |
| **R²**   | 0.99972 |


Result of optuna : 
| trial | value | n_layers | n_units | dropout_rate | lr | batch_size |
|---|---|---|---|---|---|---|
| 0.0 | 0.0005015310598537326 | 4.0 | 160.0 | 0.1086128453234521 | 0.0007452613328590502 | 16.0 |
| 1.0 | 0.002798998961225152 | 3.0 | 32.0 | 0.08494932238531852 | 0.009265240191327155 | 16.0 |
| 2.0 | 0.003490017028525472 | 4.0 | 256.0 | 0.2154438209280327 | 0.002979858531190217 | 16.0 |
| 3.0 | 0.0002837716019712389 | 2.0 | 128.0 | 0.08845483216508479 | 0.0004487403874319785 | 64.0 |
| 4.0 | 0.0004061913350597024 | 2.0 | 160.0 | 0.0413576018792639 | 0.0001724771367446595 | 64.0 |
| 5.0 | 0.001006734673865139 | 2.0 | 224.0 | 0.1112241330813663 | 0.001570799647527249 | 16.0 |
| 6.0 | 0.002282181521877646 | 4.0 | 192.0 | 0.2141724826043037 | 0.003475933577747014 | 32.0 |
| 7.0 | 0.001340325456112623 | 4.0 | 96.0 | 0.2936041115813794 | 0.0003828953334463041 | 32.0 |
| 8.0 | 0.0004733545938506722 | 2.0 | 224.0 | 0.08198468260354308 | 0.001770292795590669 | 32.0 |
| 9.0 | 0.005293113179504871 | 3.0 | 32.0 | 0.2655836258020921 | 0.006491268247871494 | 16.0 |
| 10.0 | 0.002917930483818054 | 1.0 | 96.0 | 0.380786838259487 | 0.0001015867980311987 | 64.0 |
| 11.0 | 0.0005603802273981273 | 1.0 | 128.0 | 0.01090324616342367 | 0.0001957988601862817 | 64.0 |
| 12.0 | 0.0003565321385394782 | 2.0 | 128.0 | 0.001708533191376205 | 0.0004451900676978923 | 64.0 |
| 13.0 | 0.0004475930181797594 | 2.0 | 96.0 | 0.1509443724407296 | 0.0005539607810972869 | 64.0 |
| 14.0 | 0.0004388813395053148 | 1.0 | 128.0 | 0.001209155336650745 | 0.0003651128053639437 | 64.0 |
| 15.0 | 0.00086534972069785 | 3.0 | 64.0 | 0.1636886000602692 | 0.000926080519423451 | 64.0 |
| 16.0 | 0.0003289840533398092 | 2.0 | 192.0 | 0.04772297359834911 | 0.0003188729990050884 | 64.0 |
| 17.0 | 0.001508962013758719 | 3.0 | 192.0 | 0.05033015797034853 | 0.0002002443876040518 | 64.0 |
| 18.0 | 0.0008630931843072176 | 1.0 | 192.0 | 0.1721872105439621 | 0.0002820676225569616 | 64.0 |
| 19.0 | 0.002721697557717562 | 2.0 | 256.0 | 0.1292153790026097 | 0.0001302619063058915 | 64.0 |
| 20.0 | 0.0003833064693026245 | 3.0 | 160.0 | 0.05819766230218799 | 0.001438580421284796 | 32.0 |
| 21.0 | 0.000221327441977337 | 2.0 | 128.0 | 0.02000858335541994 | 0.0005181969103804366 | 64.0 |
| 22.0 | 0.0004100569349247962 | 2.0 | 128.0 | 0.03916759995707583 | 0.0006665440262740654 | 64.0 |
| 23.0 | 0.000369979563402012 | 2.0 | 64.0 | 0.07580087367927298 | 0.0002693705543234254 | 64.0 |
| 24.0 | 0.0006259071524254978 | 1.0 | 160.0 | 0.03232057655276994 | 0.001011203896320576 | 64.0 |
| 25.0 | 0.0002896230143960565 | 2.0 | 224.0 | 0.1080555333678412 | 0.0002800994034016705 | 64.0 |
| 26.0 | 0.0002879512321669608 | 3.0 | 224.0 | 0.1215136925669772 | 0.0004860349157420588 | 64.0 |
| 27.0 | 0.0008601307636126876 | 3.0 | 96.0 | 0.1862016195255063 | 0.0005341056409202951 | 64.0 |
| 28.0 | 0.0009214126039296389 | 3.0 | 64.0 | 0.259550018527793 | 0.001012484847729272 | 32.0 |
| 29.0 | 0.0003815033996943384 | 3.0 | 160.0 | 0.1329342166225832 | 0.0006640676050633995 | 64.0 |
| 30.0 | 0.0003848991473205388 | 4.0 | 128.0 | 0.09447468756279369 | 0.0005110259058878453 | 16.0 |
| 31.0 | 0.00270864344201982 | 2.0 | 224.0 | 0.1124326900663927 | 0.0002274303840010678 | 64.0 |
| 32.0 | 0.0003933058178517967 | 2.0 | 224.0 | 0.1262116490130219 | 0.0007940772404695689 | 64.0 |
| *33.0* | *0.0002106914616888389* | *3.0* | *256.0* | *0.07481716510687292* | *0.0003997881215363161* | *64.0* |
| 34.0 | 0.0003825641178991646 | 3.0 | 256.0 | 0.07326277506100308 | 0.001300545577696934 | 64.0 |
| 35.0 | 0.0002263627684442326 | 3.0 | 256.0 | 0.02042939542692002 | 0.0003792045182975953 | 16.0 |
| 36.0 | 0.000230304867727682 | 3.0 | 256.0 | 0.02544342703903722 | 0.0003859893777433053 | 16.0 |
| 37.0 | 0.000283143570413813 | 3.0 | 256.0 | 0.01995607180008069 | 0.0001605564551178574 | 16.0 |
| 38.0 | 0.0003230632573831826 | 4.0 | 256.0 | 0.02433746278217807 | 0.0003704202608857193 | 16.0 |
| 39.0 | 0.0008478920208290219 | 3.0 | 256.0 | 0.05992642699348496 | 0.002447612249301536 | 16.0 |

so the best trail is : 33 
and we investigated : 40 difference combinations .
<p align="center">
  <img src="https://github.com/tembooo/Torque_2FG_v3/blob/main/pic4.optunaforMLP.jpg" width="700" alt="Torque vs Features">
</p>



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
