from mlflow.tracking import MlflowClient
import mlflow
import os
from src.logging_config import logger


def connect_to_daghub():
    try:
        logger.debug("Connecting to DagsHub MLflow tracking server...")

        dagshub_pat = os.getenv("DAGSHUB_PAT")

        if not dagshub_pat:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_pat
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_pat

        mlflow.set_tracking_uri(
            "https://dagshub.com/umiii-786/employee-churn-prediction.mlflow"
        )

        mlflow.set_experiment("Pipeline_RF_Model")

        logger.debug("Successfully connected to DagsHub MLflow.")

    except Exception as e:
        logger.error("Failed to connect to DagsHub MLflow tracking server.")
        raise


def promote_to_production():
    try:
        logger.debug("Promoting latest model version to production alias...")

        client = MlflowClient()

        latest_version = client.get_latest_versions("Churn_Model_With_RF")[0].version

        client.set_registered_model_alias(
            name="Churn_Model_With_RF",
            alias="production",
            version=latest_version
        )

        logger.debug(
            f"Model version {latest_version} successfully promoted to 'production' alias."
        )

    except Exception as e:
        logger.error("Failed to promote model to production alias.")
        raise


if __name__ == "__main__":
    try:
        logger.debug("Production promotion pipeline started.")

        connect_to_daghub()
        promote_to_production()

        logger.debug("Production promotion pipeline completed successfully.")

    except Exception:
        logger.critical("Pipeline failed.", exc_info=True)
        raise