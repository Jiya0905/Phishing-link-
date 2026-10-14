"""
train_model.py
---------------
This is the ACTUAL "Machine Learning" part of the project.

Steps happening here (classic ML pipeline):
1. Load a dataset (numeric features + label 0/1) - dataset.csv by default,
   or a different CSV if you pass one as an argument (e.g. a real dataset
   prepared from Kaggle via prepare_kaggle_dataset.py).
2. Split into training data (80%) and testing data (20%)
3. Train a RandomForestClassifier (a collection of decision trees that vote)
4. Test how accurate it is on data it has NEVER seen before
5. Save the trained model to model.pkl so the backend API can reuse it
   instantly without retraining every time.

Run this file AFTER generate_dataset.py (for synthetic data):
    python generate_dataset.py
    python train_model.py

Or, to train on a REAL Kaggle dataset instead:
    python prepare_kaggle_dataset.py kaggle_data/your_file.csv dataset_kaggle.csv
    python train_model.py dataset_kaggle.csv
"""

import sys
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from feature_extractor import FEATURE_NAMES

MODEL_PATH = "model.pkl"


def main():
    # Default to the synthetic dataset.csv, unless a different file is
    # given on the command line (e.g. a real Kaggle-derived dataset).
    data_path = sys.argv[1] if len(sys.argv) > 1 else "dataset.csv"

    print(f"Loading dataset from {data_path} ...")
    df = pd.read_csv(data_path)
    print(f"{len(df)} rows loaded.")

    X = df[FEATURE_NAMES]   # input features (numbers)
    y = df["label"]         # output: 0 = safe, 1 = phishing

    # 80% of data used to teach the model, 20% kept aside to test honestly
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train/test split: {len(X_train)} training rows, {len(X_test)} testing rows.")

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=200,   # 200 decision trees voting together
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("\nEvaluating model on unseen test data...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance -> which features mattered most (great for your presentation slides!)
    importances = pd.Series(model.feature_importances_, index=FEATURE_NAMES)
    importances = importances.sort_values(ascending=False)
    print("\nTop features the model relies on:")
    print(importances.head(10))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
