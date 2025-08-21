import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from loguru import logger
from pathlib import Path

sys.path.append(os.path.abspath("../src"))


def load_dataset():
    """Load, scale, and split the cancer dataset."""
    data_path = os.path.join(os.getcwd(), "data/raw/cancer_dataset.csv")
    df = pd.read_csv(data_path).drop(columns=["Unnamed: 0"])

    # Separate features and target
    X = df.drop(columns=["target"])
    y = df["target"]
    column_names = X.columns

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=column_names)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def get_root():
   root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / ".git").exists())
   return root


def preprocess_data():
    logger.info("Starting drift data generation")
    root = get_root()
    data_path = root / "data" / "cancer_data.csv"
    data = pd.read_csv(data_path) 
    logger.info("Data loaded successfully")

    y = data["target"]
    X = data.drop(columns=["target"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_drifted = X_train.copy()
    X_test_drifted = X_test.copy()
    X_train_drifted *= 1.2 
    X_test_drifted *= 1.2
    y_train_drifted = y_train.copy()
    y_test_drifted = y_test.copy()  
    logger.info("Drifted data generated successfully")

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)  
    drifted_train = pd.concat([X_train_drifted, y_train], axis=1)
    drifted_test = pd.concat([X_test_drifted, y_test], axis=1)
    
    train.to_csv(root / "data" / "train.csv", index=False)
    test.to_csv(root / "data" / "test.csv", index=False)
    drifted_train.to_csv(root / "data" / "drifted_train.csv", index=False)
    drifted_test.to_csv(root / "data" / "drifted_test.csv", index=False)
    logger.info("Drifted data saved successfully")
    
    return (X_train, X_test, y_train, y_test, X_train_drifted, y_train_drifted, X_test_drifted, y_test_drifted)
