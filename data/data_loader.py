import os
import pandas as pd


def load_data():
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, '..', 'data', 'heart.csv')
    return pd.read_csv(data_path)

def load_and_preprocess_data():
    """Load and preprocess heart dataset"""
    df = load_data()
    
    # Recode thal variable
    bins = [-1, 1, 2, 10]
    labels = ['0', '1', '2']
    df['thal'] = pd.cut(df['thal'], bins=bins, labels=labels)
    
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
    final_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'Sex', 'Cp', 'Fbs', 'Restecg', 'Exang', 'Slope', 'thal', 'Target']
    df = df.rename(columns={'thal': 'thal'})
    return df[final_cols]