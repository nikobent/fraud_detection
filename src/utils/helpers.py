import pandas as pd
import yaml
import os

def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file and return its contents as a dictionary.
    """
    full_path = os.path.join(os.getcwd(), config_path)
    with open(full_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def read_csv_data(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file from the given file path and returns a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise Exception(f"Error reading CSV file at {file_path}: {str(e)}")