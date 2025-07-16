import sys
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
sys.path.append(os.path.abspath("../src"))


def train_model(X_train, y_train):
    """Train a Random Forest model and save to pickle file."""

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    model_path = "models/model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model trained and saved to {model_path}")
