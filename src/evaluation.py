import pickle
import os
from sklearn.metrics import accuracy_score, f1_score
import mlflow


def evaluate_model(model_path, X_test, y_test):
    """
    Load the trained model from a pickle file and evaluate on the test set.

    Parameters:
    - model_path: Path to the saved .pkl model file
    - X_test: Test features
    - y_test: Test labels

    Returns:
    - accuracy: Accuracy of the model on the test set
    """


    # Load model from file
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model accuracy: {accuracy * 100:.2f}%")

    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")

    return accuracy



def ml_evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model using accuracy and F1 score,
    log metrics to MLflow, and return results.

    Parameters
    ----------
    model : sklearn estimator
        Trained model to evaluate.
    X_test : array-like
        Test feature data.
    y_test : array-like
        True labels for the test set.

    Returns
    -------
    dict
        Dictionary containing accuracy and f1_score.
    """
    # 1. Predict
    y_pred = model.predict(X_test)

    # 2. Compute metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # 3. Log metrics to MLflow
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

    return {"accuracy": acc, "f1_score": f1}
