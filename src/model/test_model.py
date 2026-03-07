import mlflow
import dagshub
import os
import pickle
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from src.logging_config import logger
import json

# DAGSHUB + MLFLOW CONFIG
dagshub.auth.add_app_token(os.environ["DAGSHUB_PAT"])
# dagshub.init(repo_owner='umiii-786', repo_name='employee-churn-prediction',mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/umiii-786/employee-churn-prediction.mlflow')

mlflow.set_experiment('Pipeline_RF_Model')


def load_model(path: str) -> RandomForestClassifier:
    try:
        logger.debug("Loading trained model...")

        load_path = os.path.join(path, 'rf_model.pkl')

        with open(load_path, 'rb') as f:
            model = pickle.load(f)

        logger.debug(f"Model loaded successfully from {load_path}")
        return model

    except Exception as e:
        logger.error(f"Error while loading model: {e}")
        raise


def load_data(path: str) -> tuple:
    try:
        logger.debug(f"Loading training and testing datasets from {path} ...")

        train_path = os.path.join(path, 'train.csv')
        test_path = os.path.join(path, 'test.csv')

        train_ds = pd.read_csv(train_path)
        test_ds = pd.read_csv(test_path)

        logger.debug(f"Train shape: {train_ds.shape}")
        logger.debug(f"Test shape: {test_ds.shape}")

        return train_ds, test_ds

    except Exception as e:
        logger.error(f"Error while loading data: {e}")
        raise


def load_yaml(path: str) -> dict:
    try:
        logger.debug("Loading model parameters from YAML file...")

        with open(path, "r") as file:
            params = yaml.safe_load(file)

        parameters = params["model_parameters"]

        logger.debug(f"Parameters loaded: {parameters}")
        return parameters

    except Exception as e:
        logger.error(f"Error while loading parameters from YAML file: {e}")
        raise


def testModel(model: RandomForestClassifier,
              parameters: dict,
              train_ds: pd.DataFrame,
              test_ds: pd.DataFrame):

    try:
        logger.debug("Model evaluation started...")

        x_train = train_ds.drop('Churn', axis=1)
        y_train = train_ds['Churn']

        x_test = test_ds.drop('Churn', axis=1)
        y_test = test_ds['Churn']

        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        with mlflow.start_run() as run:

            # Metrics
            mlflow.log_metric('train_accuracy',
                              accuracy_score(y_train, y_train_pred))

            mlflow.log_metric('test_accuracy',
                              accuracy_score(y_test, y_test_pred))

            mlflow.log_metric('train_precision',
                              precision_score(y_train, y_train_pred))

            mlflow.log_metric('test_precision',
                              precision_score(y_test, y_test_pred))

            mlflow.log_metric('train_recall',
                              recall_score(y_train, y_train_pred))

            mlflow.log_metric('test_recall',
                              recall_score(y_test, y_test_pred))

            mlflow.log_metric('train_f1',
                              f1_score(y_train, y_train_pred))

            mlflow.log_metric('test_f1',
                              f1_score(y_test, y_test_pred))

            # Parameters
            mlflow.log_params(parameters)

            # Model logging
            logged_model = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model"
            )

            model_id = logged_model.model_id

            # Log transformer artifact if exists
            if os.path.exists('models/column_transformer.pkl'):
                mlflow.log_artifact('models/column_transformer.pkl')

        logger.debug("Model evaluation completed successfully.")
        run_id=run.info.run_id
        return run_id,model_id
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

def save_run_info(runid,modelid,run_path):
    try:
        logger.debug("Saving Run Info...")
        os.makedirs('reports',exist_ok=True)
        data = {
            'run_id':runid,
            "model_id":modelid
        }

        with open(run_path, "w") as file:
            json.dump(data, file, indent=4)

        logger.debug(f"Run Info Saved Successfully from {run_path}")

    except Exception as e:
        logger.error(f"Error while Saving RunInfo: {e}")
        raise

    


def main() -> None:
    try:
        logger.debug("Model Evaluation Stage Started...")

        load_path = 'data/processed'

        model = load_model('models')
        train_ds, test_ds = load_data(load_path)
        parameters = load_yaml('params.yaml')

        run_id,model_id=testModel(model, parameters, train_ds, test_ds)
        run_path=os.path.join('reports', 'RunInfo.json')
        save_run_info(run_id,model_id,run_path)

        logger.debug("Model Evaluation Stage Completed Successfully.")

    except Exception as e:
        logger.critical(f"Pipeline failed in Evaluation stage: {e}")
        raise


if __name__ == '__main__':
    main()