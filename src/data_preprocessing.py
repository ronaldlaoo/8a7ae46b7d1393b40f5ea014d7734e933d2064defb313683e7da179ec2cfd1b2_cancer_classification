import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
