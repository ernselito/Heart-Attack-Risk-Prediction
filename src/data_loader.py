import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    """Download dataset from Kaggle"""
    api = KaggleApi()
    api.authenticate()
    os.makedirs('data', exist_ok=True)
    api.dataset_download_files(
        'johnsmith88/heart-disease-dataset',
        path=os.path.abspath('data'),
        unzip=True
    )
    print("Dataset downloaded to data/ directory")

def load_and_preprocess_data():
    """Load and preprocess heart dataset"""
    # Load data
    df = pd.read_csv('data/heart.csv')
    
    # Recode thal variable (as in R analysis)
    bins = [-1, 1, 2, 10]
    labels = ['0', '1', '2']
    df['Thal'] = pd.cut(df['thal'], bins=bins, labels=labels)
    
    # Rename columns
    df = df.rename(columns={
        'sex': 'Sex',
        'cp': 'Cp',
        'fbs': 'Fbs',
        'restecg': 'Restecg',
        'exang': 'Exang',
        'slope': 'Slope',
        'target': 'Target'
    })
    
    # Select relevant columns
    final_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 
                  'Sex', 'Cp', 'Fbs', 'Restecg', 'Exang', 'Slope', 'Thal', 'Target']
    
    return df[final_cols]