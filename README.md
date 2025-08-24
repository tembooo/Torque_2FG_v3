#part five Continuous Regression  and baseline is HistGBR
there is difference between "Continuous Regression" and "Stepwise Regression with Trees" so for this project we need to use 

| Model             | MSE     | RMSE  | MAE   | R²      |
| ----------------- | ------- | ----- | ----- | ------- |
| **HistGBR\_raw**  | 41.21   | 6.42  | 2.61  | 0.99973 |
| **HistGBR\_rule** | 38.62   | 6.21  | 2.29  | 0.99974 |
| **Linear**        | 2865.72 | 53.53 | 37.31 | 0.98092 |
| **Polynomial**    | 968.98  | 31.13 | 18.84 | 0.99355 |
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
