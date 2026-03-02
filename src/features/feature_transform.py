from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
from src.logging_config import logger
import os
import pickle

def load_data(path: str) -> tuple:
    try:
        logger.debug(f"Loading datasets from {path} ...")

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


def applyTransformation(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple:
    try:
        logger.debug("Starting feature transformation...")

        trf = ColumnTransformer(
            transformers=[
                ('standard_scaling',
                 StandardScaler(),
                 [0, 1, 2, 3, 4, 5, 6, 8]),

                ('one_hot_encoding',
                 OneHotEncoder(drop='first', sparse_output=False),
                 [7])
            ],
            remainder='passthrough'
        )

        logger.debug("Fitting transformer on training data...")
        trf.fit(x_train)

        logger.debug("Transforming training and testing data...")

        x_train_transformed = pd.DataFrame(
            trf.transform(x_train),
            columns=trf.get_feature_names_out()
        )

        x_test_transformed = pd.DataFrame(
            trf.transform(x_test),
            columns=trf.get_feature_names_out()
        )

        logger.debug("Feature transformation completed successfully.")
        logger.debug(f"Transformed train shape: {x_train_transformed.shape}")
        logger.debug(f"Transformed test shape: {x_test_transformed.shape}")

        logger.debug(f"\n Saving Column Transformer")
        pickle.dump(trf,open('models/column_transformer.pkl','wb'))
        logger.debug(f"\n saved Column Transformer")

        return x_train_transformed, x_test_transformed

    except Exception as e:
        logger.error(f"Error during feature transformation: {e}")
        raise


def save_data(train_ds: pd.DataFrame, test_ds: pd.DataFrame, save_path: str) -> None:
    try:
        logger.debug("Saving processed datasets after transformation...")

        os.makedirs(save_path, exist_ok=True)

        train_path = os.path.join(save_path, "train.csv")
        test_path = os.path.join(save_path, "test.csv")

        train_ds.to_csv(train_path, index=False)
        test_ds.to_csv(test_path, index=False)

        logger.debug(f"Datasets saved successfully at {save_path}")

    except Exception as e:
        logger.error(f"Error while saving datasets: {e}")
        raise


def main():
    try:
        logger.debug("Feature Transformation Stage Started in Pipeline...")

        load_path = 'data/interim'
        save_path = 'data/processed'

        train_ds, test_ds = load_data(load_path)

        logger.debug("Droping EmployID  Column")
        train_ds.drop('EmpId',axis=1,inplace=True)
        test_ds.drop('EmpId',axis=1,inplace=True)
        logger.debug("Separating features and target column...")

        x_train = train_ds.drop('Churn', axis=1)
        y_train = train_ds['Churn']

        x_test = test_ds.drop('Churn', axis=1)
        y_test = test_ds['Churn']

        transformed_x_train, transformed_x_test = applyTransformation(
            x_train, x_test
        )

        # Add target column back
        transformed_x_train['Churn'] = y_train.values
        transformed_x_test['Churn'] = y_test.values

        save_data(transformed_x_train, transformed_x_test, save_path)

        logger.debug("Feature Transformation Stage Completed Successfully.")

    except Exception as e:
        logger.critical(f"Pipeline failed in Feature Transformation stage: {e}")
        raise


if __name__ == '__main__':
    main()