import pandas as pd
import os
from src.logging_config import logger


def load_data(path: str) -> tuple:
    try:
        logger.info(f"Loading datasets from {path} ...")

        train_path = os.path.join(path, 'train.csv')
        test_path = os.path.join(path, 'test.csv')

        train_ds = pd.read_csv(train_path)
        test_ds = pd.read_csv(test_path)

        logger.info(f"Train shape: {train_ds.shape}")
        logger.info(f"Test shape: {test_ds.shape}")

        return train_ds, test_ds

    except Exception as e:
        logger.error(f"Error while loading data: {e}")
        raise


def replace_outliers_q1_q3(group):
    try:
        logger.debug(
            f"Handling outliers for group: {group.name}"
        )

        Q1 = group['Salary_INR'].quantile(0.25)
        Q3 = group['Salary_INR'].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        group.loc[group['Salary_INR'] < lower_bound, 'Salary_INR'] = Q1
        group.loc[group['Salary_INR'] > upper_bound, 'Salary_INR'] = Q3

        return group

    except Exception as e:
        logger.error(f"Error while handling outliers for group {group.name}: {e}")
        raise


def save_data(train_ds: pd.DataFrame, test_ds: pd.DataFrame, save_path: str) -> None:
    try:
        logger.info("Saving processed datasets after outlier handling...")

        os.makedirs(save_path, exist_ok=True)

        train_path = os.path.join(save_path, "train.csv")
        test_path = os.path.join(save_path, "test.csv")

        train_ds.to_csv(train_path, index=False)
        test_ds.to_csv(test_path, index=False)

        logger.info(f"Datasets saved successfully at {save_path}")

    except Exception as e:
        logger.error(f"Error while saving datasets: {e}")
        raise


def main():
    try:
        logger.info("Outlier Handling Pipeline Started...")

        load_path = 'data/raw'
        save_path = 'data/interim'

        train_ds, test_ds = load_data(load_path)

        logger.info("Applying group-wise IQR outlier treatment...")

        train_ds = (
            train_ds
            .groupby(['Department', 'time_spent_company'], group_keys=False)
            .apply(replace_outliers_q1_q3)
        )

        logger.info("Outlier handling completed.")

        save_data(train_ds, test_ds, save_path)

        logger.info("Outlier Handling Pipeline Completed Successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed due to: {e}")
        raise


if __name__ == '__main__':
    main()