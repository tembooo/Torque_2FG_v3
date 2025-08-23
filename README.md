
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
