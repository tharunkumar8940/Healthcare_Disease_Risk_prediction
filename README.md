# 🏥 Healthcare Disease Risk Prediction System

An end-to-end Machine Learning application that predicts disease risk based on patient health parameters.

## 📌 Project Overview

The Healthcare Disease Risk Prediction System uses machine learning techniques to analyze patient health information and estimate the probability of disease risk.

The project covers the complete machine learning workflow:

Dataset → Data Analysis → Preprocessing → Feature Engineering → Model Training → Model Evaluation → Prediction → Streamlit Application

## 🎯 Objectives

- Analyze healthcare patient data
- Perform data preprocessing and feature engineering
- Compare multiple machine learning algorithms
- Evaluate model performance using standard metrics
- Select the best-performing model
- Build an interactive disease-risk prediction application
- Provide an easy-to-use Streamlit interface

## 📊 Dataset

The project uses a healthcare dataset containing patient health parameters such as:

- Age
- BMI
- Blood Pressure
- Cholesterol
- Glucose
- Heart Rate
- Gender
- Smoking
- Physical Activity
- Family History

### Target Variable

`Disease_Risk`

The target represents the predicted disease-risk category.

## 🤖 Machine Learning Models

The following models were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. XGBoost
6. Tuned Random Forest
7. Tuned XGBoost

### 🏆 Final Model

**Tuned XGBoost Classifier**

The Tuned XGBoost model achieved the best performance among the evaluated models.

- Accuracy: 99.50%
- Precision: 99.66%
- Recall: 99.66%
- F1 Score: 99.66%
- ROC-AUC: 99.997%

## 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Plotly
- Streamlit
- Jupyter Notebook

## 🔄 Prediction Pipeline

```text
Patient Input
      ↓
Data Preprocessing
      ↓
Categorical Encoding
      ↓
Feature Alignment
      ↓
Feature Scaling
      ↓
Tuned XGBoost Model
      ↓
Risk Prediction
      ↓
Risk Probability