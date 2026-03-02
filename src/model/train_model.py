import pandas as pd
import os
from src.logging_config import logger
import yaml
from sklearn.ensemble import RandomForestClassifier
import pickle


def load_data(path: str) -> pd.DataFrame:
    try:
        logger.info(f"Loading training dataset from {path} ...")

        train_path = os.path.join(path, 'train.csv')
        train_ds = pd.read_csv(train_path)

        logger.info(f"Train shape: {train_ds.shape}")
        return train_ds

    except Exception as e:
        logger.error(f"Error while loading data: {e}")
        raise


def load_yaml(path: str) -> dict:
    try:
        logger.info("Loading model parameters from YAML file...")

        with open(path, "r") as file:
            params = yaml.safe_load(file)

        parameters = params["model_parameters"]

        logger.info(f"Parameters loaded: {parameters}")
        return parameters

    except Exception as e:
        logger.error(f"Error while loading parameters from YAML file: {e}")
        raise


def train_model(x_train: pd.DataFrame,
                y_train: pd.Series,
                parameters: dict) -> RandomForestClassifier:
    try:
        logger.info("Model training started...")

        rf = RandomForestClassifier(
            random_state=42,
            class_weight='balanced',
            **parameters
        )

        rf.fit(x_train, y_train)

        logger.info("Model training completed successfully.")
        logger.info(f"Number of trees: {rf.n_estimators}")

        return rf

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise


def save_model(model: RandomForestClassifier, path: str) -> None:
    try:
        logger.info("Saving trained model...")
        save_path = os.path.join(path, 'rf_model.pkl')

        with open(save_path, 'wb') as f:
            pickle.dump(model, f)

        logger.info(f"Model saved successfully at {save_path}")

    except Exception as e:
        logger.error(f"Error while saving model: {e}")
        raise


def main():
    try:
        logger.info("Model Training Stage Started...")

        load_path = 'data/processed'
        save_path = 'models'

        train_ds = load_data(load_path)
        parameters = load_yaml('params.yaml')

        logger.info("Separating features and target column...")

        x_train = train_ds.drop('Churn', axis=1)
        y_train = train_ds['Churn']

        trained_model = train_model(x_train, y_train, parameters)

        save_model(trained_model, save_path)

        logger.info("Model Training Stage Completed Successfully.")

    except Exception as e:
        logger.critical(f"Pipeline failed in Model Training stage: {e}")
        raise


if __name__ == '__main__':
    main()