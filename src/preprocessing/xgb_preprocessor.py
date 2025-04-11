import pandas as pd
from .base_preprocessor import BasePreprocessor


class XGBPreprocessor(BasePreprocessor):
    def transform(self, raw_data: dict) -> pd.DataFrame:
        # Apply base transformation (dropping unnecessary columns)
        df = super().transform(raw_data)

        # XGBoost-specific preprocessing:
        # Convert date strings to datetime objects (assumes "YYYYMMDD" format)
        df["claim_date_occurred"] = pd.to_datetime(df["claim_date_occurred"], format="%Y%m%d", errors="coerce")
        df["claim_date_reported"] = pd.to_datetime(df["claim_date_reported"], format="%Y%m%d", errors="coerce")

        # Create derived features:
        df["report_delay_days"] = (df["claim_date_reported"] - df["claim_date_occurred"]).dt.days
        df["car_age_at_claim"] = df["claim_date_reported"].dt.year - df["object_year_construction"]
        return df