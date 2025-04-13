import pandas as pd
from utils.helpers import load_config


class BasePreprocessor:
    def __init__(self, config_path: str = "config/preprocessor_config.yaml"):
        """
        Load configuration from the given YAML file and set up common parameters.
        """
        config = load_config(config_path)
        self.drop_cols = config.get("drop_columns", [])

    def transform(self, raw_data: dict) -> pd.DataFrame:
        """
        Convert a raw input (dictionary) to a pandas DataFrame and drop columns specified in the config.
        """
        df = pd.DataFrame([raw_data])
        if self.drop_cols[0] in df.columns:
            df = df.drop(columns=self.drop_cols, errors="ignore")
        return df