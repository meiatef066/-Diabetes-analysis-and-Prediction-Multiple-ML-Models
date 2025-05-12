# Diabetes Prediction Project

## Overview

This project utilizes machine learning to predict whether a person has diabetes or not based on the provided dataset. It implements various data preprocessing techniques, feature engineering, and classification algorithms to achieve accurate predictions.

The dataset used can be found on Kaggle: [Diabetes Dataset](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset).

---

## Dependencies

To run this project, you need to install the following libraries:

```bash
pip install lightgbm imbalanced-learn
```

Other required libraries include:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`

---

## Steps in the Project

### 1. Data Loading
We load the dataset using `pandas` and perform an initial exploration to understand its structure and contents.

### 2. Data Preprocessing
- **Handling Missing Values**: Missing values are imputed using the KNN imputation technique (`KNNImputer`).
- **Feature Scaling**: Standard scaling is applied to numerical features using `StandardScaler`.
- **Handling Imbalances**: SMOTE (Synthetic Minority Oversampling Technique) is used to handle class imbalance in the dataset.

### 3. Feature Engineering
- **Dimensionality Reduction**: PCA (Principal Component Analysis) is used to reduce the number of features if necessary.
- **Feature Selection**: `SelectKBest` with ANOVA (`f_classif`) is used to select the most important features.

### 4. Model Building
Various machine learning classifiers are implemented:
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Decision Tree**
- **Random Forest**
- **AdaBoost**
- **Gradient Boosting**
- **XGBoost**
- **LightGBM**

Each model is evaluated using:
- Accuracy
- F1-Score
- Classification Report


## Dataset

The dataset contains the following features:
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure
- **SkinThickness**: Triceps skin fold thickness
- **Insulin**: 2-Hour serum insulin
- **BMI**: Body mass index
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age in years
- **Outcome**: 0 or 1 (indicates if a person has diabetes)

---

## Results

The best model achieved the following performance metrics:
- **Accuracy**: 77%
- **F1-Score**:
   * Class 0 (Negative): 81%
   * Class 1 (Positive): 69%
- **Precision**:
   * Class 0 (Negative): 86%
   * Class 1 (Positive): 86%
- **Recall**: XX%
   * Class 0 (Negative): 77%
   * Class 1 (Positive): 76%

---

## Tools Used

- **Programming Language**: Python
- **Libraries**: scikit-learn, XGBoost, LightGBM, imbalanced-learn, pandas, numpy, matplotlib, seaborn.

---

## Contributing

Contributions are welcome! Feel free to open issues or create pull requests.
