import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

def generate_clinical_report(model, X_test, y_test):
    """
    Creates Walgreens-ready evaluation visuals
    Returns:
        dict: Clinical performance metrics
    """
    # Generate predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Create visuals
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Confusion matrix with clinical labels
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=['Low Risk', 'High Risk'],
        cmap='Blues', ax=ax[0]
    )
    ax[0].set_title("Patient Risk Classification")
    
    # ROC curve
    RocCurveDisplay.from_predictions(
        y_test, y_proba,
        name=f'Heart Attack Risk (AUC={auc:.2f})',
        ax=ax[1]
    )
    ax[1].set_title("Early Detection Performance")
    plt.savefig('../results/clinical_performance.png')
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'sensitivity': recall_score(y_test, y_pred),
        'specificity': specificity_score(y_test, y_pred)  # Custom function
    }