import unittest
import mlflow
from src.logging_config import logger
import os
# import pandas as pd
# from sklearn.metrics import accuracy_score

# Model URIs
CANDIDATE_MODEL_URI = "models:/Churn_Model_With_RF@candidate"
PRODUCTION_MODEL_URI = "models:/Churn_Model_With_RF@production"


class TestModelPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        try:
            logger.debug("Connecting to DagsHub MLflow tracking server on Testing File...")

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

        
    def test_1_candidate_model_load(self):
        """Step 1: Test candidate model can load from MLflow."""
        try:
            logger.info("Loading candidate model...")
            self.candidate_model = mlflow.pyfunc.load_model(CANDIDATE_MODEL_URI)

        except Exception as e:
            logger.error("Failed to load candidate model.", exc_info=True)
            self.fail(f"Candidate model failed to load: {e}")

        self.assertIsNotNone(self.candidate_model, "Candidate model is None after loading.")


if __name__ == "__main__":
    unittest.main()