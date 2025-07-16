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
    import pickle
    import os
    from sklearn.metrics import accuracy_score

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
