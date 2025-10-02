#!/usr/bin/env python3
"""
Credit Scoring Model
Save this file as: credit_scoring.py

Usage (examples):
  python credit_scoring.py --train
  python credit_scoring.py --predict --sample "age=30,income=50000,loan_amount=5000,loan_duration_months=24,num_credit_lines=2,delinquencies=0"

Requirements:
  pip install numpy pandas scikit-learn joblib
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib

DATA_CSV = "data/credit_data.csv"
MODEL_SAVE = "credit_scoring_model.joblib"
RANDOM_STATE = 42

def create_synthetic_dataset(path=DATA_CSV, n=5000, seed=42):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        'age': rng.randint(18, 75, size=n),
        'income': (rng.normal(40000, 15000, size=n)).astype(int),
        'loan_amount': (rng.normal(8000, 6000, size=n)).astype(int),
        'loan_duration_months': rng.randint(6, 72, size=n),
        'num_credit_lines': rng.randint(0, 10, size=n),
        'delinquencies': rng.poisson(0.5, size=n)
    })
    # Create a target that loosely depends on these fields
    score = (df['income'] / (df['loan_amount'] + 1)) + (df['age'] / 50) - df['delinquencies']*2 - df['num_credit_lines']*0.3
    prob = 1 / (1 + np.exp(-0.0001*(score*1000 - 3)))
    rng2 = np.random.RandomState(seed+1)
    df['good_credit'] = (rng2.rand(n) < prob).astype(int)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Synthetic dataset saved to {path} (shape={df.shape})")
    return df

def load_dataset(path=DATA_CSV):
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded dataset {path} shape={df.shape}")
    else:
        print(f"No dataset found at {path}. Creating a synthetic dataset.")
        df = create_synthetic_dataset(path)
    return df

def build_preprocessor(df, feature_cols):
    numeric_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, feature_cols)
    ])
    return preprocessor

def train_and_save(df, model_path=MODEL_SAVE):
    features = [c for c in df.columns if c != 'good_credit']
    X = df[features]
    y = df['good_credit']

    preprocessor = build_preprocessor(df, features)
    X_p = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_p, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    models = {
        'logreg': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'rf': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    }

    results = {}
    for name, m in models.items():
        print(f"Training {name} ...")
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        probs = m.predict_proba(X_test)[:,1] if hasattr(m, 'predict_proba') else None
        acc = accuracy_score(y_test, preds)
        roc = roc_auc_score(y_test, probs) if probs is not None else None
        results[name] = {'model': m, 'accuracy': acc, 'roc_auc': roc}
        print(f"  {name}: accuracy={acc:.4f}, roc_auc={roc}")

    # choose best by roc_auc when available
    best_name = None
    best_score = -1
    for name, r in results.items():
        score = r['roc_auc'] if r['roc_auc'] is not None else r['accuracy']
        if score > best_score:
            best_score = score
            best_name = name
    best_model = results[best_name]['model']
    print(f"Best model: {best_name} (score={best_score})")

    package = {'preprocessor': preprocessor, 'model': best_model, 'features': features}
    joblib.dump(package, model_path)
    print(f"Saved model package to {model_path}")

    # Print classification report for best model on test set
    # Re-evaluate best_model with preprocessor on test split (recompute X_test from raw X)
    # We have X_test and y_test from earlier; recompute using raw not available here so re-split:
    X_p_all = preprocessor.transform(X)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_p_all, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    preds_best = best_model.predict(X_test2)
    print("Classification report (best model):")
    print(classification_report(y_test2, preds_best, zero_division=0))

def load_package(model_path=MODEL_SAVE):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model package not found at {model_path}. Train first.")
    pkg = joblib.load(model_path)
    return pkg['preprocessor'], pkg['model'], pkg['features']

def predict_sample(sample_dict, model_path=MODEL_SAVE):
    preprocessor, model, features = load_package(model_path)
    # Build a single-row DataFrame in correct column order
    row = {f: sample_dict.get(f, 0) for f in features}
    df_row = pd.DataFrame([row])
    Xs = preprocessor.transform(df_row)
    pred = model.predict(Xs)[0]
    prob = model.predict_proba(Xs)[:,1][0] if hasattr(model, 'predict_proba') else None
    return pred, prob

def parse_sample_string(s):
    # Expects "age=30,income=50000,loan_amount=5000,loan_duration_months=24,num_credit_lines=2,delinquencies=0"
    items = s.split(',')
    d = {}
    for it in items:
        if '=' in it:
            k, v = it.split('=', 1)
            k = k.strip()
            v = v.strip()
            try:
                d[k] = float(v) if '.' in v else int(v)
            except:
                d[k] = v
    return d

def main():
    parser = argparse.ArgumentParser(description="Credit Scoring model script")
    parser.add_argument('--train', action='store_true', help='Train model (uses data/credit_data.csv or synthetic)')
    parser.add_argument('--predict', action='store_true', help='Predict single sample')
    parser.add_argument('--sample', type=str, default=None, help='Sample string for prediction (k=v,...)')
    parser.add_argument('--model-path', type=str, default=MODEL_SAVE, help='Path to save/load model package')
    args = parser.parse_args()

    if args.train:
        df = load_dataset(DATA_CSV)
        train_and_save(df, model_path=args.model_path)
    elif args.predict:
        if not args.sample:
            print("Provide --sample 'k=v,k2=v2,...' for prediction")
            return
        sample_dict = parse_sample_string(args.sample)
        pred, prob = predict_sample(sample_dict, model_path=args.model_path)
        print(f"Prediction (1=good credit): {pred}, Prob(good): {prob}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
