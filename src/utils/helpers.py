import pandas as pd

def read_csv_data(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file from the given file path and returns a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise Exception(f"Error reading CSV file at {file_path}: {str(e)}")