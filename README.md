# Python-Project-2026
Adult Income Prediction – End-to-End Machine Learning Project

## Overview

This project builds a machine learning pipeline to predict whether an individual earns more than $50K per year based on demographic and employment data.

The goal is not just model accuracy, but to demonstrate **clean data workflows, reproducibility, and business-focused insights** using modern best practices.

## Business Problem

Understanding income distribution is valuable for:

* Targeted marketing
* Credit risk assessment
* Customer segmentation
* Policy and socio-economic analysis

This model predicts high-income individuals (>50K), enabling better decision-making in these areas.


## Dataset

* Source: UCI Adult Income Dataset
* Records: ~32,000
* Features include:

  * Age
  * Workclass
  * Education
  * Marital Status
  * Occupation
  * Hours per week
  * Capital gain/loss
  * Gender, Race

## Target variable:

* Income → Binary classification (<=50K or >50K)

## Approach

### 1. Data Preprocessing

* Handled missing values (`?` → NaN)
* Separated numerical and categorical features
* Applied:

  * Median imputation (numeric)
  * Most frequent imputation (categorical)
  * One-hot encoding for categorical variables
  * Standard scaling for numeric features

All preprocessing is handled using a **Pipeline + ColumnTransformer** to ensure:

* No data leakage
* Reproducibility
* Clean, maintainable code


### 2. Model

* **Random Forest Classifier**
* Integrated into full pipeline
* Evaluated using cross-validation


### 3. Validation Strategy

* Train/Test split (80/20)
* Stratified K-Fold Cross Validation (5 folds)
* Focus on **ROC-AUC** (better for imbalanced datasets)


Results

Metric   Score (example) 
ROC-AUC  ~0.90           
Accuracy | ~85%

(Actual results may vary slightly depending on random state)


Evaluation

* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)
* ROC-AUC Score

This ensures a **balanced view of performance**, not just accuracy.

Key Features

* End-to-end ML pipeline
* No data leakage (Pipeline used correctly)
* Handles mixed data types cleanly
* Reproducible workflow
* Industry-standard evaluation metrics
* Ready for deployment extension


Key Learnings

* Importance of pipelines in production ML
* Handling categorical data correctly (OneHotEncoder vs LabelEncoder)
* Why **ROC-AUC is better than accuracy** for classification problems
* Structuring ML projects for **real-world use**



Tech Stack

* Python
* Pandas / NumPy
* Scikit-learn
* Seaborn / Matplotlib

---

Future Improvements

* Hyperparameter tuning (RandomizedSearchCV)
* Try gradient boosting models (XGBoost / LightGBM)
* Model explainability (SHAP values)
* Deploy model via API (Flask / FastAPI)

This project is part of my portfolio demonstrating skills in:

* Data analysis
* Machine learning
* Building production-ready pipelines
