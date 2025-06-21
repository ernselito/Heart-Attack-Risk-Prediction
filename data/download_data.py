import os
import subprocess
from kaggle.api.kaggle_api_extended import KaggleApi

def download_clinical_dataset():
    """Downloads heart dataset from Kaggle with Mac path handling"""
    api = KaggleApi()
    api.authenticate()
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Download dataset
    api.dataset_download_files(
        'johnsmith88/heart-disease-dataset',
        path=os.path.abspath('data'),
        unzip=True
    )
    print(f"✅ Clinical dataset downloaded to: {os.path.abspath('data')}")

if __name__ == "__main__":
    download_clinical_dataset()