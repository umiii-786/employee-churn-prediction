import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import yaml
from src.logging_config import logger


def load_data(data_name: str, path: str) -> pd.DataFrame:
    try:
        logger.debug("Starting dataset loading from KaggleHub...")
        
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            path,
            data_name,
        )

        logger.debug(f"Dataset loaded successfully. Shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error while loading dataset: {e}")
        raise


def load_yaml(path: str) -> float:
    try:
        logger.debug("Loading parameters from YAML file...")

        with open(path, "r") as file:
            params = yaml.safe_load(file)

        test_size = params["data_ingestion"]["test_size"]

        logger.debug(f"Test size loaded from YAML: {test_size}")
        return test_size

    except Exception as e:
        logger.error(f"Error while loading YAML file: {e}")
        raise


def save_data(train_ds: pd.DataFrame, test_ds: pd.DataFrame) -> None:
    try:
        logger.debug("Saving train and test datasets...")

        os.makedirs("data/raw", exist_ok=True)

        train_path = os.path.join("data/raw", "train.csv")
        test_path = os.path.join("data/raw", "test.csv")

        train_ds.to_csv(train_path, index=False)
        test_ds.to_csv(test_path, index=False)

        logger.debug("Train and Test datasets saved successfully.")

    except Exception as e:
        logger.error(f"Error while saving datasets: {e}")
        raise


def main() -> None:
    try:
        logger.debug("Data Ingestion Started...")

        data_name = "Employee_HR.csv"
        path = "prishatank/employee-hr-dataset"
        yaml_path = "params.yaml"

        df = load_data(data_name, path)
        test_size = load_yaml(yaml_path)

        logger.debug("Splitting dataset into train and test...")
        train_ds, test_ds = train_test_split(
            df, test_size=test_size, random_state=4
        )

        logger.debug("Dataset split completed.")
        save_data(train_ds, test_ds)

        logger.debug("Data Ingestion Completed Successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed due to: {e}")
        raise


if __name__ == "__main__":
    main()