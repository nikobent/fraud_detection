import unittest
import pandas as pd
from src.preprocessing.xgb_preprocessor import XGBPreprocessor
from src.preprocessing.llm_preprocessor import LLMPreprocessor
from src.utils.helpers import load_config


class TestPreprocessors(unittest.TestCase):
    def setUp(self):
        # Full raw data example (simulate a claim row with all necessary columns)
        self.raw_data = {
            "sys_sector": "Private NonLife",
            "sys_label": "FRISS",
            "sys_process": "Claims_initial_load",
            "sys_product": "MOTOR",
            "sys_dataspecification_version": "4.5",
            "sys_claimid": "MTR-338957796-02",
            "sys_currency_code": "EUR",
            "claim_amount_claimed_total": 2433,
            "claim_causetype": "Collision",
            "claim_date_occurred": "20121022",
            "claim_date_reported": "20121127",
            "claim_location_urban_area": 1,
            "object_make": "VOLKSWAGEN",
            "object_year_construction": 2008,
            "ph_firstname": "Kostas",
            "ph_gender": "F",
            "ph_name": "Papadopoulos",
            "policy_fleet_flag": "0",
            "policy_insured_amount": 74949,
            "policy_profitability": "Low"
        }

    def test_xgb_preprocessor(self, config_path: str = "config/preprocessor_config.yaml"):
        preproc = XGBPreprocessor()
        df = preproc.transform(self.raw_data)

        # Check the dropped columns are absent, after loading them
        config = load_config(config_path)
        drop_cols = config.get("drop_columns", [])
        for col in drop_cols:
            self.assertNotIn(col, df.columns)

        # Check that date conversion worked and the derived features exist
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["claim_date_occurred"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["claim_date_reported"]))
        self.assertIn("report_delay_days", df.columns)
        self.assertIn("car_age_at_claim", df.columns)

    def test_llm_preprocessor(self, config_path: str = "config/preprocessor_config.yaml"):
        preproc = LLMPreprocessor()
        df = preproc.transform(self.raw_data)

        # The LLM preprocessor should only keep the narrative columns
        config = load_config(config_path)
        expected_columns = config.get("story_columns", [])

        self.assertListEqual(list(df.columns), expected_columns)

        # Test the natural language sentence generation
        sentence = preproc.to_sentence(df)
        self.assertIsInstance(sentence, str)
        self.assertGreater(len(sentence), 0)

if __name__ == '__main__':
    unittest.main()
