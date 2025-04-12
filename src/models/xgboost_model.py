# src/models/xgb_model.py
import joblib
from .base_model import FraudModel

class XGBFraudModel(FraudModel):
    def __init__(self, pipeline):
        """
        pipeline: a scikit-learn Pipeline that includes preprocessing (e.g., ColumnTransformer) and the XGBoost classifier.
        """
        self.pipeline = pipeline

    def fit(self, X, y):
        self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)

    def save(self, path: str):
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str):
        pipeline = joblib.load(path)
        return cls(pipeline)
