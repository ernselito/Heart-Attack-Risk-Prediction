# app.py
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import os

# Load model and data with robust path handling
@st.cache_resource
def load_model():
    # Try different paths for local vs cloud deployment
    model_paths = [
        'model/knn_model.pkl',              # Local development path
        '../model/knn_model.pkl',           # Streamlit Cloud path
        'src/model/knn_model.pkl'           # Alternative structure
    ]
    for path in model_paths:
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError("Model file not found in any expected locations")

@st.cache_data
def load_data():
    # Try different paths for local vs cloud deployment
    data_paths = [
        'data/heart.csv',                   # Local development path
        '../data/heart.csv',                # Streamlit Cloud path
        'src/data/heart.csv'                # Alternative structure
    ]
    for path in data_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError("Dataset not found in any expected locations")

# Initialize
try:
    model = load_model()
    df = load_data()
    st.session_state.model_loaded = True
except Exception as e:
    st.error(f"Error loading resources: {str(e)}")
    st.session_state.model_loaded = False
    st.stop()

# App title and description
st.title('❤️ Heart Disease Risk Predictor')
st.markdown("""
Interactive tool for predicting heart disease risk using machine learning.
Adjust patient parameters on the left to see prediction results.
""")

# Sidebar for inputs
st.sidebar.header('Patient Parameters')

# Input widgets with validation
ca = st.sidebar.slider('Major Vessels (0-3)', 0, 3, 1)
cp = st.sidebar.selectbox('Chest Pain Type', 
                         options=[0, 1, 2, 3],
                         format_func=lambda x: 
                         ['Typical Angina', 'Atypical Angina', 
                          'Non-anginal Pain', 'Asymptomatic'][x])
sex = st.sidebar.radio('Gender', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
oldpeak = st.sidebar.slider('ST Depression', 0.0, 6.0, 1.0)
thal = st.sidebar.selectbox('Thalassemia', 
                           options=[0, 1, 2],
                           format_func=lambda x: 
                           ['Normal', 'Fixed Defect', 'Reversible Defect'][x])
thalach = st.sidebar.slider('Max Heart Rate', 70, 200, 150)
exang = st.sidebar.radio('Exercise Angina', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

# Create feature vector with correct data types
input_data = [[ca, cp, sex, oldpeak, str(thal), thalach, exang]]
columns = ['ca', 'Cp', 'Sex', 'oldpeak', 'Thal', 'thalach', 'Exang']

# Make prediction
if st.sidebar.button('Predict Risk'):
    input_df = pd.DataFrame(input_data, columns=columns)
    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        risk_prob = proba[1]  # Probability of heart disease
        
        # Display results
        st.subheader('Prediction Result')
        if prediction == 1:
            st.error(f'High Risk of Heart Disease ({risk_prob:.1%} probability)')
            st.markdown("**Recommendation:** Immediate cardiology referral")
        else:
            st.success(f'Low Risk of Heart Disease ({1-risk_prob:.1%} probability)')
            st.markdown("**Recommendation:** Routine follow-up")
        
        # Show probability gauge
        st.progress(float(risk_prob))
        st.caption(f'Risk probability: {risk_prob:.1%}')
    
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# EDA Visualizations
st.header('Data Exploration')
tab1, tab2, tab3 = st.tabs(["Correlation", "Feature Distributions", "Model Performance"])

with tab1:
    st.subheader('Correlation Matrix')
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)

with tab2:
    st.subheader('Feature vs Target')
    feature = st.selectbox('Select feature', 
                         ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'])
    fig, ax = plt.subplots()
    sns.boxplot(x='Target', y=feature, data=df, ax=ax)
    st.pyplot(fig)

with tab3:
    st.subheader('Model Performance')
    try:
        # Try multiple paths for prediction data
        pred_paths = [
            'model/prediction_data.csv',
            '../model/prediction_data.csv',
            'src/model/prediction_data.csv'
        ]
        pred_data = None
        for path in pred_paths:
            if os.path.exists(path):
                pred_data = pd.read_csv(path)
                break
        
        if pred_data is None:
            st.warning("Performance data not found")
        else:
            fpr, tpr, _ = roc_curve(pred_data['y_test'], pred_data['y_prob'])
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic')
            ax.legend(loc="lower right")
            st.pyplot(fig)
            
    except Exception as e:
        st.warning(f"Could not display performance data: {str(e)}")

# Dataset preview
st.header('Dataset Preview')
st.dataframe(df.head(10))
st.caption(f'Full dataset contains {len(df)} patient records')