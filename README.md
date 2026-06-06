# 🏥 Medical Insurance Cost Prediction

A supervised machine learning project to predict annual medical insurance charges based on patient demographics and lifestyle factors.

---

## 📌 Project Overview

This project applies multiple regression algorithms to predict insurance costs using a dataset of 1,338 records. The workflow covers the complete ML pipeline from data exploration to model deployment.

---

## 📂 Dataset

- **Source:** Medical Insurance Dataset
- **Records:** 1,338
- **Features:** Age, Sex, BMI, Children, Smoker, Region
- **Target:** Annual Insurance Charges (USD)

---

## 🔧 Steps Covered

| Step | Description |
|------|-------------|
| 1 | Import Libraries & Load Data |
| 2 | Exploratory Data Analysis (EDA) |
| 3 | Missing Values & Outlier Treatment |
| 4 | Feature Engineering & Preprocessing |
| 5 | Model Building — Multiple Regressors |
| 6 | Model Evaluation & Overfitting Check |
| 7 | Hyperparameter Tuning |
| 8 | Final Model Comparison + Streamlit App |

---

## 🤖 Models Used

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor ⭐ (Best)
- AdaBoost Regressor
- Support Vector Regressor (SVR)
- K-Nearest Neighbors (KNN)
- XGBoost Regressor

---

## 📊 Best Model Performance

| Metric | Score |
|--------|-------|
| Model | Gradient Boosting (Tuned) |
| Test R² | ~0.8655 |
| MAE | ~$2,035.4 |

---

## 🚀 Streamlit App

🔗 **Live App:** [Medical Insurance Cost Predictor](https://medical-insurance-cost-predicter.streamlit.app/)

Enter patient details → Get estimated annual insurance cost instantly.

---

## 🛠️ Tech Stack

- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn
- Streamlit

---

## 👤 Author

**Vineeth Muraleedharan**  
Senior Radiation Therapist | Healthcare AI Enthusiast  
E&ICT Academy, IIT Guwahati - Applied Data Science, ML & AI (2025–26)

---
