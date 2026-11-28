"""Shared cleaning code for the CS4650 capstone (Kaggle Playground S6E10).

Everyone imports from here so every model sees exactly the same data:

    from src.clean import load_data, make_features, make_split

Facts about the raw data (checked on train.csv, 699,635 rows):
  * Target `satisfaction` is True/False (44.4% True). Score = ROC AUC.
  * Only missing values: 292 in `Arrival Delay in Minutes` (130 in test).
  * 13 service ratings on a 0-5 scale. 0 means "not rated / not applicable",
    not "terrible", so treating ratings as categories helps linear models.
  * Delays and flight distance are heavily right-skewed -> log1p.
  * 4 text columns: Gender, Customer Type, Type of Travel, Class.
  * No duplicate rows.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TARGET = "satisfaction"
ID = "id"
CATEGORICAL = ["Gender", "Customer Type", "Type of Travel", "Class"]
RATINGS = [
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]
SKEWED = ["Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]


def load_data(data_dir=DATA_DIR):
    """Return (train, test) DataFrames, with the target as 0/1."""
    train = pd.read_csv(Path(data_dir) / "train.csv")
    test = pd.read_csv(Path(data_dir) / "test.csv")
    train[TARGET] = train[TARGET].astype(int)
    return train, test


def _clean(df):
    df = df.copy()
    # Missing arrival delay: arrival and departure delay correlate at 0.84,
    # so copy departure delay into the gap.
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(
        df["Departure Delay in Minutes"]
    )
    for col in SKEWED:
        df[col] = np.log1p(df[col])
    return df


def make_features(train, test, ratings_as_categories=True):
    """Return X_train, y_train, X_test (all numeric) with identical columns.

    ratings_as_categories=True one-hot encodes the 0-5 ratings. That lifted
    logistic regression from 0.921 to 0.948 AUC on our validation split.
    Tree models don't need it; pass False for them.
    """
    y = train[TARGET]
    X = _clean(train.drop(columns=[ID, TARGET]))
    X_test = _clean(test.drop(columns=[ID]))

    one_hot = CATEGORICAL + (RATINGS if ratings_as_categories else [])
    X = pd.get_dummies(X, columns=one_hot, dtype=float)
    X_test = pd.get_dummies(X_test, columns=one_hot, dtype=float)
    X_test = X_test.reindex(columns=X.columns, fill_value=0.0)
    return X, y, X_test


def make_split(X, y):
    """The team's one validation split. Always use this so scores compare."""
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def save_submission(test, probabilities, name):
    """Write a Kaggle-ready CSV (id, satisfaction probability)."""
    root = Path(__file__).resolve().parent.parent
    out = root / "submissions" / f"{name}.csv"
    pd.DataFrame({ID: test[ID], TARGET: probabilities}).to_csv(out, index=False)
    return f"saved {out.relative_to(root)}"
