from data_ingestion import ingest_data
from data_preprocessing import preprocess_data
from model_training import ml_train_model
from evaluation import ml_evaluate_model
from drift_detection import detect_drift
from loguru import logger
import mlflow


def main():
    """Full pipeline: load data, train, evaluate, and register model if threshold met."""
    mlflow.set_tracking_uri("http://localhost:5000")

    ingest_data()
    logger.info("Data saved as csv")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_drifted,
        y_train_drifted,
        X_test_drifted,
        y_test_drifted,
    ) = preprocess_data()
    logger.info("Data split completed")

    with mlflow.start_run(run_name="rf-breast-cancer") as run:
        # Train
        model = ml_train_model(X_train, y_train)

        # Evaluate
        metrics = ml_evaluate_model(model, X_test, y_test)
        
        # Baseline drift: train vs test
        baseline_drift_results = detect_drift("data/train.csv", "data/test.csv")
        mlflow.log_param("baseline_drift_detected", baseline_drift_results["drift_detected"])
        mlflow.log_metric("baseline_overall_drift_score", baseline_drift_results["overall_drift_score"])

        if baseline_drift_results["drift_detected"]:
            raise ValueError("Baseline data drift detected (train vs test)! Retraining required.")
        
        # Threshold check (classification: accuracy > 0.8)
        if metrics["accuracy"] > 0.8:
            mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name="breast_cancer_rf",
            )
            logger.info("Model registered")
        else:
            logger.info("Model did not meet threshold")

        # Drifted drift: test vs drifted_test
        drifted_drift_results = detect_drift("data/test.csv", "data/drifted_test.csv")
        mlflow.log_param("drifted_drift_detected", drifted_drift_results["drift_detected"])
        mlflow.log_metric("drifted_overall_drift_score", drifted_drift_results["overall_drift_score"])

        if drifted_drift_results["drift_detected"]:
            raise ValueError("Drift detected in DRIFTED test set! Retraining required.")
        
if __name__ == "__main__":
    main()
    logger.info("Pipeline completed successfully.")
