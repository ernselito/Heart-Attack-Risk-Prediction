# Heart Disease Prediction - Machine Learning Project

## Project Overview
This project evaluates the performance of various machine learning algorithms for predicting heart disease diagnosis. The study compares multiple classification models using a comprehensive dataset of 1,025 patient records with 14 clinical features.

##  Objective
To identify the most effective machine learning model for heart disease prediction through comparative analysis, hyperparameter tuning, and performance evaluation.

##  Dataset
- **Source**: Kaggle - "Heart Disease Dataset" by johnsmith88
- **Size**: 1,025 records with 14 features
- **Target Variable**: Binary classification (0 = no disease, 1 = disease)
- **No Missing Values**: Complete dataset with no null values

### Features Description:
- **Demographic**: age, sex
- **Medical History**: cp (chest pain type), trestbps (resting blood pressure), chol (cholesterol)
- **Test Results**: fbs (fasting blood sugar), restecg (resting electrocardiographic results)
- **Exercise-related**: thalach (maximum heart rate), exang (exercise induced angina), oldpeak (ST depression)
- **Other Medical**: slope, ca (number of major vessels), thal (thalassemia)

##  Methodology

### 1. Data Preprocessing
- **Data Splitting**: 75-25 train-test split
- **Feature Engineering**:
  - Age binning into 6 groups
  - One-hot encoding for categorical variables
  - Standard scaling for numerical features
- **Preprocessing Pipeline**: ColumnTransformer for handling different feature types

### 2. Models Evaluated
- Logistic Regression
- Linear Discriminant Analysis (LDA)
- K-Nearest Neighbors (KNN)
- Decision Trees
- Random Forest
- Support Vector Machines (SVMs)
- Gradient Boosting

### 3. Model Selection Process
- Initial accuracy evaluation
- Cross-validation (5-fold) for robustness assessment
- Hyperparameter tuning using GridSearchCV for top performers

##  Results

### Initial Model Performance
| Model | Accuracy | Cross-Val Mean | Cross-Val Std |
|-------|----------|----------------|---------------|
| Random Forest | 98.5% | 98.2% | 0.016 |
| Decision Trees | 97.1% | 97.6% | 0.019 |
| KNN (k=3) | 95.1% | 90.9% | 0.028 |
| Gradient Boosting | 93.2% | 93.8% | 0.026 |
| LDA | 82.4% | 86.7% | 0.021 |
| Logistic Regression | 81.5% | 87.1% | 0.028 |

### Hyperparameter Tuning Results
- **K-Nearest Neighbors**: Best Score - 98.9%
  - Optimal parameters: n_neighbors=3, p=2, weights='distance'
- **Random Forest**: Best Score - 98.4%
- **Decision Trees**: Best Score - 97.9%

### Final Model Performance
After hyperparameter tuning, the optimized KNN model achieved:
- **Test Accuracy**: 100%
- **Confusion Matrix**: Perfect classification (102 true negatives, 103 true positives)
- **Classification Report**: All metrics (precision, recall, F1-score) at 1.00

##  Key Findings

1. **Top Performing Models**: Random Forest, Decision Trees, and KNN demonstrated the highest accuracy
2. **Best Model**: K-Nearest Neighbors with optimized parameters achieved perfect prediction
3. **Feature Importance**: Strong correlations observed between target and features like chest pain type (cp), maximum heart rate (thalach), and exercise-induced angina (exang)

##  Insights & Recommendations

- **Clinical Application**: The high accuracy suggests potential for clinical decision support systems
- **Model Robustness**: Cross-validation confirmed model stability across different data splits
- **Feature Importance**: Chest pain type and exercise test results are strong predictors
- **Future Work**: Explore deep learning approaches and additional clinical features

##  Conclusion
This project successfully demonstrates that machine learning models, particularly K-Nearest Neighbors with proper parameter tuning, can achieve exceptional performance in heart disease prediction. The results highlight the potential of ML in medical diagnostics and provide a robust framework for similar healthcare prediction tasks.

---
*Note: This project is for educational and research purposes. Always consult healthcare professionals for medical diagnoses.*
