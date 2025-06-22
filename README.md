#  Heart Disease Risk Prediction System

**93.2% accurate clinical risk stratification model** that identifies high-risk patients using 7 key clinical indicators. 

![Project Pipeline](https://img.shields.io/badge/Pipeline-Data_Processing→EDA→Model_Training→App_Deployment-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Key Performance Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 93.2% |
| Sensitivity | 91.6% |
| Specificity | 94.8% |
| AUC | 0.932 |

## Features
- Interactive risk prediction dashboard
- Comprehensive model evaluation (1000 trials)
- Clinical decision support with recommendations
- Data exploration visualizations
- One-click reproduction of results

## Installation
```bash
git clone https://github.com/richkaitoo/Heart-Attack-Risk-Prediction.git
cd heart-disease-prediction

# Install dependencies
pip install -r requirements.txt

# Download dataset (requires Kaggle API setup)
python src/data_loader.py
```