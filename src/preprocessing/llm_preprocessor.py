import pandas as pd
from .base_preprocessor import BasePreprocessor
from utils.helpers import load_config



class LLMPreprocessor(BasePreprocessor):
    def transform(self,  raw_data: dict, config_path: str = "config/preprocessor_config.yaml") -> pd.DataFrame:
        # Apply base transformation
        df = super().transform(raw_data)
        # For LLM processing, we keep only the columns needed to build a narrative.
        config = load_config(config_path)

        columns_for_narrative = config.get("story_columns", [])
        df = df[columns_for_narrative]
        return df

    def to_sentence(self, df: pd.DataFrame) -> str:
        """
        Convert the processed DataFrame (assumed to have one row) into a human-readable sentence.
        """
        row = df.iloc[0]
        # string similar to what was embedded into the vector DB
        return (
            f"A {row['object_make']} car manufactured in {row['object_year_construction']} had a "
            f"{row['claim_causetype']} claim of {row['claim_amount_claimed_total']} EUR, "
            f"which occurred on {row['claim_date_occurred']} and was reported on {row['claim_date_reported']}. "
            f"Policy profitability: {row['policy_profitability']}, fleet flag: {row['policy_fleet_flag']}."
        )
