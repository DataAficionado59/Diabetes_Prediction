# Diabetes Prediction using Machine Learning

This project implements a machine learning model to predict whether a person has diabetes or not based on various medical parameters. It uses the PIMA Indians Diabetes dataset, which contains several medical predictor variables and one target variable (whether a person has diabetes).

## Project Overview

The notebook walks through the following steps:

1. **Importing Libraries**:
    - The necessary libraries such as `numpy`, `pandas`, and `scikit-learn` for data manipulation and model training are imported.
  
2. **Data Collection and Analysis**:
    - The dataset is loaded from a CSV file and basic analysis is performed.
    - The dataset used in this project is the PIMA Diabetes dataset, which consists of several features such as:
      - `Pregnancies`
      - `Glucose`
      - `BloodPressure`
      - `SkinThickness`
      - `Insulin`
      - `BMI`
      - `DiabetesPedigreeFunction`
      - `Age`
      - `Outcome` (Target variable, indicating whether a person has diabetes or not)

3. **Preprocessing**:
    - Data is cleaned, standardized, and split into training and testing sets.
    - Features are scaled using `StandardScaler` for better model performance.

4. **Model Training**:
    - A Support Vector Machine (SVM) classifier is trained using the `scikit-learn` library.
    - The model is trained on the training dataset and evaluated using the test dataset.

5. **Evaluation**:
    - The performance of the model is evaluated based on accuracy using `accuracy_score`.

## Requirements

To run this notebook, you need the following libraries installed:

- Python 3.x
- NumPy
- Pandas
- Scikit-learn

You can install these packages using pip:

```bash
pip install numpy pandas scikit-learn
