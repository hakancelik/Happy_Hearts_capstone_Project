# Heart Disease Prediction Project

## Project Overview

This project involves building and evaluating several machine learning models for predicting heart disease based on a dataset containing patient health information. The main objective is to preprocess the data, perform feature engineering, and train various classification models to identify the best-performing one.

## Dataset

The dataset used is `heart.csv`, which includes features such as age, cholesterol levels, and heart rate metrics, and is aimed at predicting the presence of heart disease in patients.

## Features Engineering

1. **Age to Max Heart Rate Ratio**: `age_max_heart_rate_ratio` = age / max heart rate
2. **Age Range**: Categorized into bins [30-39, 40-49, 50-59, 60-69, 70-79]
3. **Cholesterol to Max Heart Rate Ratio**: `cholesterol_hdl_ratio` = cholesterol / max heart rate
4. **Heart Rate Reserve**: `heart_rate_reserve` = max heart rate - resting blood pressure

## Dependencies

Ensure you have the following packages installed:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib joblib streamlit
```
https://happyheartscapstonemiuul.streamlit.app/
