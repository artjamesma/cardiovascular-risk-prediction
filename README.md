# Cardiovascular Risk Prediction Using Machine Learning

An end-to-end machine-learning project for predicting cardiovascular disease using demographic, clinical, and lifestyle data.

This project demonstrates a complete machine-learning workflow from exploratory data analysis and reproducible data cleaning through model comparison, cross-validation, threshold analysis, SHAP explainability, and model serialization.

> **Important:** This project is intended for educational and analytical purposes. The resulting model has not been clinically validated and should not be used as a medical diagnostic system.

---

## Project Overview

The objective of this project is to develop and evaluate machine-learning models capable of predicting the presence of cardiovascular disease from patient characteristics.

The original dataset contains **70,000 patient records**. After reproducible data-quality filtering, **68,754 observations** were retained for modeling.

The project includes:

- Exploratory Data Analysis (EDA)
- Data-quality assessment and cleaning
- BMI feature engineering
- Logistic Regression modeling
- Random Forest modeling
- XGBoost modeling
- Five-fold cross-validation
- Hyperparameter tuning
- Model comparison and selection
- ROC and confusion-matrix analysis
- Classification-threshold analysis
- Global SHAP explainability
- Individual prediction explanations
- SHAP dependence analysis
- Model serialization and verification
- Reproducibility metadata

---

## Dataset

The target variable is `cardio`:

- `0` — No cardiovascular disease
- `1` — Cardiovascular disease

The final model uses **12 predictor variables**:

| Feature | Description |
|---|---|
| `age` | Patient age |
| `gender` | Gender |
| `height` | Height |
| `weight` | Weight |
| `ap_hi` | Systolic blood pressure |
| `ap_lo` | Diastolic blood pressure |
| `cholesterol` | Cholesterol category |
| `gluc` | Glucose category |
| `smoke` | Smoking status |
| `alco` | Alcohol consumption |
| `active` | Physical activity status |
| `bmi` | Body Mass Index |

### Data Preparation

The raw dataset contained **70,000 observations**.

After applying reproducible data-quality rules, **68,754 observations** remained.

The cleaned modeling dataset was divided using a stratified 80/20 train-test split:

- **Training observations:** 55,003
- **Test observations:** 13,751

Stratification was used to preserve the cardiovascular-disease class distribution in both subsets.

---

## Models Evaluated

Three primary classification algorithms were evaluated:

1. **Logistic Regression**
2. **Random Forest**
3. **XGBoost**

Models were evaluated using multiple performance measures rather than accuracy alone:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Classification Report
- Five-fold cross-validation

XGBoost produced the strongest overall discrimination and was selected as the final model.

---

## Final Model Performance

### XGBoost Classifier

| Metric | Test Result |
|---|---:|
| **Accuracy** | **73.45%** |
| **Precision** | **75.51%** |
| **Recall** | **68.60%** |
| **F1 Score** | **71.89%** |
| **ROC-AUC** | **0.8022** |

The ROC-AUC score of approximately **0.80** indicates useful discriminatory ability on the held-out test set.

The model correctly balances predictive performance across several metrics, although the appropriate classification threshold depends on the relative importance assigned to false-positive and false-negative predictions.

---

## Cross-Validation and Hyperparameter Tuning

Five-fold cross-validation was used to assess model stability beyond a single train-test split.

XGBoost was also evaluated using randomized hyperparameter optimization.

The tuned configuration did not materially outperform the baseline XGBoost model during cross-validation. Therefore, the simpler baseline XGBoost configuration was retained as the final model.

This decision avoids unnecessary model complexity when additional tuning does not provide meaningful improvement.

---

## Classification Threshold Analysis

The default classification threshold of **0.50** was compared with thresholds ranging from **0.30 to 0.70**.

The analysis demonstrates the expected precision-recall trade-off:

- Lower thresholds increase **recall**, detecting more positive cases but generating more false positives.
- Higher thresholds increase **precision**, but produce more false negatives.
- The default threshold of **0.50** provides a reasonable overall balance for this analytical project.

Threshold selection should ultimately depend on the intended application and the relative cost of false-positive and false-negative predictions.

---
## Model Explainability with SHAP

SHAP (SHapley Additive exPlanations) was used to interpret how the final XGBoost model generates predictions.

The explainability analysis was performed at two levels:

### Global Explainability

Global SHAP analysis identifies the variables that have the greatest overall influence on model predictions.

The most influential features were:

1. **Systolic blood pressure (`ap_hi`)**
2. **Age**
3. **Cholesterol**
4. **Diastolic blood pressure (`ap_lo`)**
5. **BMI**

Systolic blood pressure was the strongest overall contributor to model predictions.

SHAP dependence plots were also used to investigate how changes in important features affect model output. In particular, increasing systolic blood pressure and increasing age generally shifted predictions toward higher modeled cardiovascular risk.

### Individual Prediction Explanation

SHAP was also used to explain individual predictions.

For an example high-risk observation, the model predicted a cardiovascular-disease probability of approximately **94.11%**.

The strongest contributor to this prediction was elevated systolic blood pressure, with BMI, weight, cholesterol, and diastolic blood pressure also contributing to the model's output.

This demonstrates how model predictions can be decomposed into feature-level contributions rather than treating the classifier entirely as a black box.

---

## Key Findings

The analysis produced several important findings:

- XGBoost achieved the strongest overall predictive discrimination among the evaluated models.
- The final model achieved a **ROC-AUC of 0.8022** on the held-out test set.
- Systolic blood pressure was the most influential feature in the SHAP analysis.
- Age and cholesterol were also major contributors to model predictions.
- Classification threshold selection substantially affects the balance between precision and recall.
- Hyperparameter tuning did not meaningfully improve XGBoost cross-validation performance, supporting retention of the simpler baseline configuration.
- SHAP analysis provided both global and patient-level explanations of model behavior.

---

## Saved Model Artifacts

The final trained model and supporting metadata are saved as:

```text
cardio_xgboost_model.pkl
cardio_model_metadata.json

---

## Project Structure

```text
cardio-risk-prediction/
│
├── 01_Cardio_Risk_Data_Understanding.ipynb
├── 02_Cardio_Risk_Final_Project.ipynb
│
├── health_data.csv
│
├── cardio_xgboost_model.pkl
├── cardio_model_metadata.json
│
├── README.md
├── requirements.txt
├── LICENSE
│
└── Supporting documentation and earlier experimental notebooks