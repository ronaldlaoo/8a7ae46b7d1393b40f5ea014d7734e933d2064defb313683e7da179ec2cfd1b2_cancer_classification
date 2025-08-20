import sys
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import joblib
import pandas as pd

sys.path.append(os.path.abspath("../src"))


def train_model(X_train, y_train):
    """Train a Random Forest model and save to pickle file."""

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    model_path = "models/model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model trained and saved to {model_path}")




class CustomMLModel(mlflow.pyfunc.PythonModel):
    """Custom MLflow PyFunc wrapper for your trained model."""
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None

    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model"])
        if "preprocessor" in context.artifacts:
            self.preprocessor = joblib.load(context.artifacts["preprocessor"])
        if "feature_names" in context.artifacts:
            with open(context.artifacts["feature_names"], "r") as f:
                self.feature_names = [ln.strip() for ln in f.readlines()]

    def predict(self, context, model_input: pd.DataFrame):
        X = (
            self.preprocessor.transform(model_input)
            if self.preprocessor is not None
            else model_input.values
        )
        return self.model.predict(X)


def ml_train_model(X_train, y_train):
    """
    Train a RandomForest with grid search, log hyperparameters,
    save artifacts, and log a custom PyFunc model to MLflow.
    """

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10],
        "random_state": [1, 42],
    }

    base = RandomForestClassifier()
    grid = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    best_params = {
        "n_estimators": grid.best_params_["n_estimators"],
        "max_depth": grid.best_params_["max_depth"],
        "random_state": grid.best_params_["random_state"],
    }
    mlflow.log_params(best_params)
    mlflow.log_metric("cv_best_score", float(grid.best_score_))

    # save local artifacts
    os.makedirs("models", exist_ok=True)
    model_pickle_path = "models/model.pkl"
    with open(model_pickle_path, "wb") as f:
        pickle.dump(model, f)

    feature_names_path = "models/feature_names.txt"
    cols = getattr(X_train, "columns", None)
    with open(feature_names_path, "w") as f:
        if cols is not None:
            f.write("\n".join(map(str, cols)))

    mlflow.log_artifact(model_pickle_path)
    mlflow.log_artifact(feature_names_path)

    mlflow.sklearn.log_model(model, artifact_path="model")
    mlflow.pyfunc.log_model(
        artifact_path="pyfunc_model",
        python_model=CustomMLModel(),
        artifacts={
            "model": model_pickle_path,
            "feature_names": feature_names_path,
            # "preprocessor": "models/preprocessor.pkl",  
        },
    )

    print(f"Best params: {best_params} | model saved to {model_pickle_path}")
    return model