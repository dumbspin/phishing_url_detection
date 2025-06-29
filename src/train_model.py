import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import xgboost as xgb
import joblib
import os
import matplotlib.pyplot as plt
from .feature_extraction import extract_url_features, extract_url_features_bulk, fetch_whois_for_domains, get_whois_features

def train_url_classifier(features_csv='data/url_features.csv', model_path='models/url_xgb_model.joblib'):
    # Load features and labels
    df = pd.read_csv(features_csv)
    if 'label' not in df.columns:
        raise ValueError('The features CSV must contain a "label" column.')
    # Select only numeric columns for training (excluding 'url' and 'label')
    feature_cols = [col for col in df.columns if col not in ['url', 'label'] and pd.api.types.is_numeric_dtype(df[col])]
    X = df[feature_cols]
    y = df['label']

    # Check for at least two unique classes
    if y.nunique() < 2:
        raise ValueError('The label column must contain at least two unique classes for classification.')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Compute scale_pos_weight for class imbalance
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    if n_positive == 0:
        scale_pos_weight = 1
    else:
        scale_pos_weight = n_negative / n_positive

    # Train XGBoost classifier with scale_pos_weight
    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight)
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print('Accuracy:', acc)
    print('Precision:', prec)
    print('Recall:', rec)
    print('F1 Score:', f1)
    print('ROC AUC Score:', roc_auc)
    print('Confusion Matrix:\n', cm)
    print('\nClassification Report:\n', classification_report(y_test, y_pred))

    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    # Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f'Model saved to {model_path}')

if __name__ == "__main__":
    train_url_classifier() 