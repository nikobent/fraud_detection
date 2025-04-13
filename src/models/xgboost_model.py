import joblib
import os
import yaml
from .base_model import FraudModel


class XGBFraudModel(FraudModel):
    def __init__(self, pipeline):
        """
        pipeline: a scikit-learn Pipeline that includes preprocessing (e.g., ColumnTransformer)
                  and the XGBoost classifier.
        """
        self.pipeline = pipeline

    @classmethod
    def from_config(cls, config_path: str = "config/xgboost_config.yaml"):
        full_path = os.path.join(os.getcwd(), config_path)
        try:
            with open(full_path, "r") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"Failed to load config from {config_path}: {str(e)}")

        model_path = config.get("path", "models/xgboost/best_xgboost_model.pkl")
        # If model_path is a list, pick the first element.
        if isinstance(model_path, list):
            model_path = model_path[0]
        return model_path

    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)

    def save(self, path: str):
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls):
        model_path = cls.from_config()
        pipeline = joblib.load(model_path)
        return cls(pipeline)
