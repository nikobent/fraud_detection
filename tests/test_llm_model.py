# tests/test_llm_model.py
import unittest
import os
import yaml
import pandas as pd
from src.models.llm_model import LLMFraudModel


class DummyFaissIndex:
    def search(self, query_embedding, k):
        # For testing, just return indices 0...k-1 with dummy distances.
        import numpy as np
        return np.zeros((1, k)), [list(range(k))]


class TestLLMFraudModel(unittest.TestCase):
    def setUp(self):
        # Create a temporary config for testing.
        self.temp_config = {
            "model_name": "google/flan-t5-base",
            "temperature": 0.7,
            "max_new_tokens": 150,
            "prompt_prefix": "You are an expert in motor insurance fraud detection. Fraud cases are extremely rare (below 1%).",
            "num_retrievals": 5
        }
        self.temp_config_path = "temp_llm.yaml"
        with open(self.temp_config_path, "w") as f:
            yaml.dump(self.temp_config, f)
        self.llm_model = LLMFraudModel.from_config(self.temp_config_path)
        # Monkey-patch local pipeline to simulate output.
        self.llm_model.llm = lambda prompt: [{"generated_text": "Yes, fraudulent."}]
        # Create dummy FAISS index and metadata.
        self.dummy_index = DummyFaissIndex()
        # Dummy metadata must include a "text" field and a "label" field.
        self.dummy_metadata = [{"text": f"Dummy example {i}: sample narrative.", "label": 0} for i in range(10)]

    def test_build_prompt(self):
        query = "Test claim narrative."
        global_examples = ["Global example 1 (Label: Fraud)", "Global example 2 (Label: Non-Fraud)"]
        retrieved_examples = ["Retrieved example 1 (Label: Non-Fraud)", "Retrieved example 2 (Label: Fraud)"]
        prompt = self.llm_model.build_prompt(query, global_examples, retrieved_examples)
        self.assertIn(query, prompt)
        #self.assertIn("Retrieved Example", prompt)

    def test_full_predict(self):
        data = {"query": "Test claim narrative."}
        # Create a dummy CSV with at least 5 rows (with 'label' column)
        temp_df = pd.DataFrame({
            "label": [1, 0, 0, 0, 0],
            "sys_dataspecification_version": ["4.5"] * 5,
            "sys_claimid": [f"ID_{i}" for i in range(5)],
            "claim_amount_claimed_total": [2000, 2100, 2200, 2300, 2400],
            "claim_causetype": ["Collision"] * 5,
            "claim_date_occurred": ["20120101"] * 5,
            "claim_date_reported": ["20120110"] * 5,
            "claim_location_urban_area": [1] * 5,
            "object_make": ["Toyota"] * 5,
            "object_year_construction": [2010] * 5,
            "policy_fleet_flag": ["0"] * 5,
            "policy_profitability": ["Low"] * 5,
            "report_delay_days": [9] * 5,
            "car_age_at_claim": [0] * 5
        })
        temp_csv = "temp_test.csv"
        temp_df.to_csv(temp_csv, index=False)
        prediction = self.llm_model.full_predict(data, temp_csv, self.dummy_index, self.dummy_metadata, k_retrieval=5)
        self.assertEqual(prediction, 1)
        os.remove(temp_csv)

    def tearDown(self):
        if os.path.exists(self.temp_config_path):
            os.remove(self.temp_config_path)


if __name__ == "__main__":
    unittest.main()
