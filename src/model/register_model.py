from mlflow.tracking import MlflowClient
import mlflow
from src.logging_config import logger
import json
import os
import dagshub

dagshub_pat=os.getenv("DAGSHUB_PAT")
if not dagshub_pat:
    raise EnvironmentError('DAGSHUB_PAT environment variable is not setted ') 

os.environ['MLFLOW_TRACKING_USERNAME']=dagshub_pat 
os.environ['MLFLOW_TRACKING_PASSWORD']=dagshub_pat 

mlflow.set_tracking_uri('https://dagshub.com/umiii-786/employee-churn-prediction.mlflow')
client = MlflowClient()

print(mlflow.get_tracking_uri())
def load_run_info(run_path: str) -> dict:
    try:
        logger.debug("Loading Run Info...")

        if not os.path.exists(run_path):
            raise FileNotFoundError(f"{run_path} does not exist")

        with open(run_path, "r") as file:   # FIXED: read mode
            run_info = json.load(file)

        if "run_id" not in run_info:
            raise KeyError("run_id not found in RunInfo.json")

        logger.debug(f"Run Info loaded successfully from {run_path}")
        return run_info

    except Exception as e:
        logger.error("Error while loading RunInfo")
        raise


def register_model(model_name: str, run_info: dict) -> None:
    try:
        logger.debug("Registering model to MLflow Registry...")

        # model_uri = f"runs:/{run_info['run_id']}/{run_info['model_path']}"
        model_uri = f"models:/{run_info['model_id']}"
        print(model_uri)

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        logger.debug(
            f"Model registered successfully. Version: {model_version.version}"
        )

        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=model_version
        )

        logger.debug(
            f"Model {model_name} version {model_version.version} "
            f"transitioned to Production stage."
        )

    except Exception as e:
        logger.error("Error while registering model")
        raise


def main() -> None:
    try:
        logger.info("Model registration pipeline started")

        run_path = "reports/RunInfo.json"
        model_name = "Churn_Model_With_RF"

        run_info = load_run_info(run_path)
        print(run_info)
        register_model(model_name, run_info)

        logger.debug("Model registration pipeline completed successfully")

    except Exception:
        logger.critical("Pipeline failed", exc_info=True)
        raise


if __name__ == "__main__":
    main()