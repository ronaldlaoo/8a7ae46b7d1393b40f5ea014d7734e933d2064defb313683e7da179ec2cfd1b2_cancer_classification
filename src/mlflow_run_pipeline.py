from data_ingestion import ingest_data
from data_preprocessing import preprocess_data
from model_training import ml_train_model
from evaluation import ml_evaluate_model
from loguru import logger
import mlflow

def main():
    """Full pipeline: load data, train, evaluate, and register model if threshold met."""
    mlflow.set_tracking_uri("http://localhost:5000")

    # 1. Load dataset
    ingest_data() 
    logger.info("Data saved as csv")
    X_train, X_test, y_train, y_test, X_train_drifted, y_train_drifted, X_test_drifted, y_test_drifted = preprocess_data()
    logger.info("Data split completed")

    # 2. Train
    model = ml_train_model(X_train, y_train)

    # 3. Evaluate
    metrics = ml_evaluate_model(model, X_test, y_test)

    # 4. Check threshold (classification: accuracy > 0.8)
    if metrics["accuracy"] > 0.8:
        with mlflow.start_run(run_name="rf-breast-cancer") as run:
            # Register model
            mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name="breast_cancer_rf"
            )
            print("Model registered ✅")
    else:
        print("Model did not meet threshold, skipping registration ❌")

if __name__ == "__main__":
    main()
    logger.info("Pipeline completed successfully.")