# tests/test_xgb_model.py
import unittest
import pandas as pd
from src.models.xgboost_model import XGBFraudModel
from src.utils.helpers import load_config
import joblib
import os

class TestXGBFraudModel(unittest.TestCase):
    def setUp(self, config_path: str = "config/xgboost_config.yaml"):
        # Assume the model is saved as models/v1_xgb_model.pkl from training.
        config = load_config(config_path)
        model_path = config.get("path", [])[0]

        if os.path.exists(model_path):
            self.model = XGBFraudModel.load(model_path)
        else:
            self.model = None

    def test_predict(self):
        if self.model is None:
            self.skipTest("XGB model not available for testing.")
        # Simulated preprocessed input that matches the pipeline schema:
        test_df = pd.DataFrame({
            "sys_dataspecification_version": ["4.5"],
            "sys_claimid": ["MTR-338957796-02"],
            "claim_amount_claimed_total": [2433],
            "claim_causetype": ["Collision"],
            "claim_date_occurred": [pd.to_datetime("20121022", format="%Y%m%d")],
            "claim_date_reported": [pd.to_datetime("20121127", format="%Y%m%d")],
            "claim_location_urban_area": [1],
            "object_make": ["VOLKSWAGEN"],
            "object_year_construction": [2008],
            "policy_fleet_flag": ["0"],
            "policy_profitability": ["Low"],
            "report_delay_days": [36],  # Derived from dates
            "car_age_at_claim": [4]
        })
        prediction = self.model.predict(test_df)
        print(prediction[0])
        self.assertIn(prediction[0], [0, 1])

if __name__ == '__main__':
    unittest.main()
