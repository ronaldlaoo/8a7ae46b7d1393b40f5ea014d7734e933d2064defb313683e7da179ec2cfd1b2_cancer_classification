from data_preprocessing import load_dataset
from feature_engineering import add_features
from model_training import train_model
from evaluation import evaluate_model


def main():
    # Load dataset
    X_train, X_test, y_train, y_test = load_dataset()
    print("Dataset loaded successfully.")
    # Add features
    X_train = add_features(X_train)
    X_test = add_features(X_test)

    # Train model
    train_model(X_train, y_train)

    print("Model training completed.")
    # Evaluate model
    model_path = "models/model.pkl"
    evaluate_model(model_path, X_test, y_test)


if __name__ == "__main__":
    main()
