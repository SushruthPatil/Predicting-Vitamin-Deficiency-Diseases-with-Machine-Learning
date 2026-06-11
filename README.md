# Vitamin Deficiency Disease Diagnosis - ML Classification

This repository contains my work on Assignment 2 for my Masters degree at UTS as a data science student. I was provided with a patient dataset containing clinical measurements, dietary intake records, lifestyle information and symptom flags, and worked through the full machine learning pipeline from EDA to final model evaluation.

The assignment required at least 4 experiments. I ran 6, testing Decision Tree, Random Forest, HistGradientBoosting, XGBoost, Logistic Regression with SMOTE and SVM with SMOTE. Algorithms were limited to what is available in scikit-learn, with XGBoost added as an extension.

---

## Business Problem

Healthcare providers have no fast, reliable way to screen patients for vitamin deficiency-related diseases without extensive lab work. This model takes a patient's dietary intake, clinical measurements and reported symptoms as input and predicts which of five diagnostic categories they fall into: Healthy, Anemia, Rickets/Osteomalacia, Night Blindness or Scurvy.

The prediction is used to recommend a personalised vitamin intake plan. Missing a disease diagnosis is far more harmful than a false positive, so recall on minority classes, particularly Scurvy (176 patients) and Night Blindness (269 patients) was the primary clinical concern throughout all experiments.

---

## Dataset

5800 patient records with 45 columns including demographic information, lifestyle factors, vitamin intake percentages as a proportion of the Recommended Dietary Allowance, blood serum measurements and binary symptom flags. Data was provided as part of UTS course assessment and cannot be shared publicly.

**Target variable:** `disease_diagnosis` - 5 classes with a 12.1x imbalance between Healthy (2121) and Scurvy (176).

---

## Approach

### Data Cleaning
721 rows were dropped from the original 5800, leaving 5079 clean records (87.6% retained):
- 271 rows with negative values in serum and vitamin intake columns - biologically impossible data entry errors
- 369 rows with Unknown entries in categorical columns - investigation confirmed these rows had missing values across multiple other columns simultaneously, indicating low quality records
- 81 rows overlap between the two groups

### Feature Selection
22 features selected from 45 columns using 5 approaches: visual EDA, correlation analysis, ANOVA F-statistic, chi-square test and business relevance assessment. Key drops included age, bmi and hemoglobin_levels (no class separation), exercise_level and smoking_status (flat distributions across all classes) and alcohol_consumption (34% missing).

### Feature Engineering
7 engineered features created across experiments:
- Disease-specific symptom scores combining clinically related binary symptoms
- Interaction features capturing joint risk factors (diet type × sun exposure, vegan diet × B12 intake)
- Vitamin D × calcium product for bone disease risk
- Experiment-specific risk indices for Scurvy and Night Blindness

### Data Splitting
Stratified 70/15/15 train/validation/test split with `random_state=42` throughout. Stratification ensures class proportions are preserved across all three splits given the severe class imbalance.

---

## Results

| Experiment | Model | Val Macro F1 | Outcome |
|---|---|---|---|
| Baseline | DummyClassifier (most frequent) | 0.1073 | — |
| Experiment 1 | Decision Tree + GridSearchCV | 0.6809 | Partially Confirmed |
| Experiment 2 | Random Forest + RandomizedSearchCV | 0.7329 | Partially Confirmed |
| Experiment 3 | HistGradientBoosting + RandomizedSearchCV | **0.7647** | Confirmed |
| Experiment 4 | XGBoost + RandomizedSearchCV | 0.7638 | Partially Confirmed |
| Experiment 5 | Logistic Regression + SMOTE | 0.5770 | Not Confirmed |
| Experiment 6 | SVM (RBF) + SMOTE | 0.5990 | Not Confirmed |

**Final model:** HistGradientBoosting from Experiment 3, selected on highest validation macro F1 of 0.7647 and smallest cross-val to validation gap.

**Test set result:** Macro F1 of 0.7625 - a gap of just 0.002 from validation, confirming the model generalises reliably to unseen data.

---

## Key Findings

- Tree-based models significantly outperformed linear models. The vitamin deficiency classification problem is non-linear and tree ensembles handled this naturally
- SMOTE did not help. Logistic Regression with SMOTE scored 0.5770 and SVM with SMOTE scored 0.5990, both below all tree-based results. The problem is model capacity, not just class imbalance
- Scurvy remained the hardest class across all experiments, peaking at 0.52 F1. The 12.1x imbalance and high overlap with Anemia made it consistently difficult
- The strongest individual features were vitamin_c_intake, vitamin_a_intake and serum_vitamin_d — all with direct clinical links to specific disease classes
- Engineered symptom score features contributed meaningfully, particularly rickets_symptom_score and scurvy_symptom_score

---

## Project Structure

```
notebooks/
    a_EDA.ipynb               — Business objective, dataset understanding, feature exploration
    b_Preparation.ipynb       — Feature selection, data cleaning, feature engineering, train/val/test split
    c_Baseline.ipynb          — DummyClassifier baseline, metric justification
    d_experiment_1.ipynb      — Decision Tree with GridSearchCV
    e_experiment_2.ipynb      — Random Forest with RandomizedSearchCV
    f_experiment_3.ipynb      — HistGradientBoosting with RandomizedSearchCV
    g_experiment_4.ipynb      — XGBoost with RandomizedSearchCV
    h_experiment_5.ipynb      — Logistic Regression with SMOTE
    i_experiment_6.ipynb      — SVM (RBF) with SMOTE + Final model test evaluation

report/
    project_report.docx       — Full written report

data/
    disease.csv               — Not included (UTS assessment data)

README.md
```

---

## Tech Stack

Python, pandas, scikit-learn, xgboost, imbalanced-learn, numpy, matplotlib, seaborn, altair

---

## Academic Integrity

This project was completed as part of UTS 36106 Machine Learning Algorithms and Applications assessment. The code and report are shared for portfolio purposes only. Please do not copy or submit any part of this work as your own. Academic misconduct is taken seriously at UTS and all universities.

---

## Author

Sushruta Gangadhar Patil
Master of Data Science and Innovation
University of Technology Sydney
