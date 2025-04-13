import pandas as pd
import yaml
import os
import pickle
import faiss

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


def load_faiss_db(db_dir: str):
    """
        load Faiss index and metadata
    """
    index_path = os.path.join(db_dir, "claim_index.faiss")
    metadata_path = os.path.join(db_dir, "claim_metadata.pkl")
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise Exception("FAISS index or metadata not found.")
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

"""
def load_model(path:str , model_type:str):

    if model_type == "llm":
        try:
            model = LLMFraudModel.from_config("config/llm.yaml")
        except Exception as e:
            llm_model = None

    else:
        try:
            model = XGBFraudModel.load(path)
        except Exception as e:
            model = None
"""