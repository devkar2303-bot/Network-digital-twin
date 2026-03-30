"""
Network Digital Twin — ML Core
Trains a Random Forest to predict link failures 5 minutes ahead.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
from sklearn.preprocessing import StandardScaler
import joblib


FEATURES = [
    'utilisation', 'latency_ms', 'packet_loss',
    'error_rate', 'hour_of_day', 'is_core_link', 'capacity_mbps',
]
TARGET = 'failure'


def load_data(path='data/synthetic_logs.csv'):
    df = pd.read_csv(path)
    # Lag features: previous sample's utilisation 
    df = df.sort_values(['link', 'timestamp'])
    df['util_lag1']  = df.groupby('link')['utilisation'].shift(1).fillna(0)
    df['err_lag1']   = df.groupby('link')['error_rate'].shift(1).fillna(0)
    return df


def train(df):
    feat_cols = FEATURES + ['util_lag1', 'err_lag1']
    X = df[feat_cols].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight='balanced',   # handles class imbalance
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        'f1':       round(f1_score(y_test, y_pred), 4),
        'roc_auc':  round(roc_auc_score(y_test, y_prob), 4),
        'cm':       confusion_matrix(y_test, y_pred).tolist(),
        'report':   classification_report(y_test, y_pred, output_dict=True),
    }

    # Feature importance
    fi = dict(zip(feat_cols, clf.feature_importances_.round(4)))
    metrics['feature_importance'] = dict(
        sorted(fi.items(), key=lambda x: -x[1])
    )

    Path('results').mkdir(exist_ok=True)
    joblib.dump(clf,    'results/twin_model.pkl')
    joblib.dump(scaler, 'results/scaler.pkl')
    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("=== Digital Twin Training Results ===")
    print(f"F1-score : {metrics['f1']}")
    print(f"ROC-AUC  : {metrics['roc_auc']}")
    print("\nConfusion matrix (TN FP / FN TP):")
    print(np.array(metrics['cm']))
    print("\nTop-3 features:")
    for k, v in list(metrics['feature_importance'].items())[:3]:
        print(f"  {k:20s}  {v:.4f}")

    return clf, scaler, metrics


def predict_live(sample: dict):
    """
    Predict failure probability for a single link sample (dict).
    Used by the dashboard for real-time inference.
    """
    feat_cols = FEATURES + ['util_lag1', 'err_lag1']
    clf    = joblib.load('results/twin_model.pkl')
    scaler = joblib.load('results/scaler.pkl')
    row    = np.array([[sample.get(f, 0) for f in feat_cols]])
    row_s  = scaler.transform(row)
    prob   = clf.predict_proba(row_s)[0, 1]
    return round(float(prob), 4)


if __name__ == '__main__':
    df = load_data()
    train(df)
