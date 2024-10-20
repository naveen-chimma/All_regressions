# All_Regressions

## Machine Learning Classification & Regression Models

This project implements various machine learning classification and regression models using the **50_Startups** dataset. The goal is to predict profit categories for startups based on their R&D Spend, Administration, Marketing Spend, and State. The models include Linear Regression, Logistic Regression, K-Nearest Neighbors (KNN), Naive Bayes, Decision Tree Classifier, and Random Forest Classifier.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [License](#license)
-  [Requirements](#requirements)

## Introduction

The Python script in this repository trains and evaluates several machine learning models on the **50_Startups.csv** dataset. This dataset contains information about various startups, including their expenditures and profit. The target variable is **Profit**, which is categorized into three groups: low (0), medium (1), and high (2).

## Features

- **Data loading and preprocessing**:
  - Automatically loads and cleans the dataset.
  - Maps the categorical 'State' column to numerical values.
  - Categorizes profit into three classes.
  
- **Model training and evaluation**:
  - Includes six machine learning models for both classification and regression tasks.
  - Outputs detailed metrics for each model:
    - Accuracy
    - Loss (for regression)
    - Confusion Matrix
    - Classification Report
## Models Implemented

The script includes the following machine learning models:
1. **Linear Regression**:
   - Measures R-squared score and Mean Squared Error (MSE) for both training and testing sets.

2. **Logistic Regression**:
   - Evaluates classification accuracy, confusion matrix, and classification report.

3. **K-Nearest Neighbors (KNN)**:
   - Outputs accuracy, loss, confusion matrix, and classification report.

4. **Naive Bayes (GaussianNB)**:
   - Provides classification accuracy, confusion matrix, and classification report.

5. **Decision Tree Classifier**:
   - Generates classification metrics such as accuracy, confusion matrix, and classification report.

6. **Random Forest Classifier**:
   - Trains the model with different numbers of estimators (trees) and evaluates performance using accuracy, confusion matrix, and classification report.

## Results

The script will output the following metrics for each model:
- **Accuracy**: Overall performance of the model on training and testing data.
- **Loss (for regression models)**: Calculated as Mean Squared Error (MSE).
- **Confusion Matrix**: A matrix showing the performance of the classification models.
- **Classification Report**: Precision, recall, and F1-score for each class.

## Example Output for Random Forest Classifier:
------ Random Forest Classifier ------

Number of estimators: [11]

Train accuracy: 1.0000

Test accuracy:1.0000

Train classification report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        49
           1       1.00      1.00      1.00        54
           2       1.00      1.00      1.00        39

    accuracy                           1.00       142
   macro avg       1.00      1.00      1.00       142
weighted avg       1.00      1.00      1.00       142

Train confusion matrix:

[[49  0  0]

 [ 0 54  0]
 
 [ 0  0 39]]
 
Test classification report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00        17
           2       1.00      1.00      1.00         9

    accuracy                           1.00        36
   macro avg       1.00      1.00      1.00        36
weighted avg       1.00      1.00      1.00        36

Test confusion matrix:

[[10  0  0]

 [ 0 17  0]
 
 [ 0  0  9]]

-----------------------------------

## License

This project is licensed under the MIT License - see the LICENSE file for details.

You can copy this entire block and paste it into your GitHub repository for your project documentation!

## Requirements

To run the script, you need the following Python libraries:

```bash
pandas
scikit-learn
numpy


You can install these using:
pip install pandas scikit-learn numpy

