import unittest
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score


class TestModelPipeline(unittest.TestCase):

    def test_1_load_model_and_transformer(self):
        """Test whether model and transformer load successfully"""

        try:
            with open("./models/model.pkl", "rb") as f:
                model = pickle.load(f)

            with open("./models/column_transformer.pkl", "rb") as f:
                transformer = pickle.load(f)

        except Exception as e:
            self.fail(f"Loading model or transformer failed: {e}")

        self.assertIsNotNone(model)
        self.assertIsNotNone(transformer)


    def test_2_load_holdout_data(self):
        """Test loading of holdout dataset"""

        try:
            data = pd.read_csv("data/interim/test.csv")
        except Exception as e:
            self.fail(f"Loading holdout dataset failed: {e}")

        self.assertGreater(len(data), 0, "Holdout dataset is empty")
        self.assertIn("target", data.columns, "Target column missing in dataset")


    def test_3_model_performance(self):
        """Test if model accuracy is >= 70%"""

        # Load model and transformer
        with open("models/rf_model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("models/column_transformer.pkl", "rb") as f:
            transformer = pickle.load(f)

        # Load holdout data
        data = pd.read_csv("data/interim/test.csv")

        X = data.drop("Churn", axis=1)
        y = data["Churn"]

        # Transform features
        X_transformed = transformer.transform(X)

        # Predict
        predictions = model.predict(X_transformed)

        # Evaluate
        accuracy = accuracy_score(y, predictions)

        print(f"\nModel Accuracy: {accuracy}")

        self.assertGreaterEqual(
            accuracy,
            0.70,
            f"Model accuracy {accuracy} is below 70%"
        )


if __name__ == "__main__":
    unittest.main()