import os
import time
import numpy as np
import pickle
import faiss
from sklearn.metrics import f1_score, precision_score, recall_score

# Import your models – adjust the paths if needed.
from models.xgboost_model import XGBFraudModel
from models.llm_model import LLMFraudModel

# Import your LLM preprocessor (to convert raw rows into narratives)
from preprocessing.llm_preprocessor import LLMPreprocessor


def load_faiss_db(db_dir: str):
    """
    Load the FAISS index and its metadata from the specified directory.
    """
    index_path = os.path.join(db_dir, "claim_index.faiss")
    metadata_path = os.path.join(db_dir, "claim_metadata.pkl")
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise Exception("FAISS index or metadata file not found in the directory: " + db_dir)
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def compare_models(test_csv: str,
                   train_csv: str,
                   xgb_model_path: str,
                   llm_config_path: str,
                   faiss_db_dir: str,
                   sample_size: int = 1000,
                   k_retrieval: int = 5):
    """
    Compare predictions of the XGBoost and LLM models on the first sample_size test instances.
    Measures F1-score, precision, recall (for fraud vs. non-fraud), and average latency per model.

    Args:
        test_csv (str): Path to the test CSV.
        train_csv (str): Path to the training CSV (for global example sampling).
        xgb_model_path (str): Path to the pickled XGBoost model.
        llm_config_path (str): Path to the LLM configuration file (YAML).
        faiss_db_dir (str): Directory containing the FAISS index and metadata.
        sample_size (int): Number of test instances to evaluate (default 1000).
        k_retrieval (int): Number of retrieval examples for the LLM (default 5).

    Returns:
        dict: A dictionary with computed metrics and latencies for both models.
    """
    # Load test set.
    import pandas as pd

    test_df = pd.read_csv(test_csv)
    if len(test_df) > sample_size:
        test_df = test_df.iloc[:sample_size]
    true_labels = test_df["sys_fraud"].values

    # Load XGB model.
    try:
        xgb_model = XGBFraudModel.load(xgb_model_path)
    except Exception as e:
        print("Error loading XGB model:", e)
        xgb_model = None

    # Load LLM model.
    try:
        llm_model = LLMFraudModel.from_config(llm_config_path)
    except Exception as e:
        print("Error loading LLM model:", e)
        llm_model = None

    # Load FAISS index and metadata.
    try:
        faiss_index, metadata = load_faiss_db(faiss_db_dir)
    except Exception as e:
        print("Error loading FAISS database:", e)
        faiss_index, metadata = None, None

    # Prepare lists to store predictions and latencies.
    xgb_preds = []
    llm_preds = []
    xgb_latencies = []
    llm_latencies = []

    # For each row in the test set, get predictions from both models.
    # Instantiate LLMPreprocessor once (ideally caching its configuration).
    preproc = LLMPreprocessor()

    for idx, row in test_df.iterrows():
        row_dict = row.to_dict()
        # XGB prediction (assume XGB expects a DataFrame with all columns).
        if xgb_model is not None:
            import pandas as pd
            start = time.time()
            try:
                # Depending on how your xgb_model was trained,
                # you might need to supply the full row in a DataFrame.
                xgb_input = pd.DataFrame([row_dict])
                xgb_pred = xgb_model.predict(xgb_input)[0]
            except Exception as e:
                print(f"XGB prediction error for instance {idx}: {e}")
                xgb_pred = 0
            xgb_latencies.append(time.time() - start)
            xgb_preds.append(xgb_pred)
        else:
            xgb_preds.append(0)
            xgb_latencies.append(0)

        # LLM prediction.
        if llm_model is not None and faiss_index is not None and metadata is not None:
            try:
                # Convert the raw row to a narrative query.
                transformed = preproc.transform(row_dict)
                query = preproc.to_sentence(transformed)
            except Exception as e:
                print(f"LLM preprocessor error for instance {idx}: {e}")
                query = ""
            start = time.time()
            input_data = {"query": query}
            try:
                llm_pred = llm_model.full_predict(input_data, train_csv, faiss_index, metadata, k_retrieval=k_retrieval)
            except Exception as e:
                print(f"LLM prediction error for instance {idx}: {e}")
                llm_pred = 0
            llm_latencies.append(time.time() - start)
            llm_preds.append(llm_pred)
        else:
            llm_preds.append(0)
            llm_latencies.append(0)

    # Compute performance metrics.
    metrics = {}
    if len(xgb_preds) > 0:
        metrics["xgb"] = {
            "f1_score": f1_score(true_labels, xgb_preds),
            "precision": precision_score(true_labels, xgb_preds, zero_division=0),
            "recall": recall_score(true_labels, xgb_preds, zero_division=0),
            "avg_latency": np.mean(xgb_latencies)
        }
    if len(llm_preds) > 0:
        metrics["llm"] = {
            "f1_score": f1_score(true_labels, llm_preds),
            "precision": precision_score(true_labels, llm_preds, zero_division=0),
            "recall": recall_score(true_labels, llm_preds, zero_division=0),
            "avg_latency": np.mean(llm_latencies)
        }

    return metrics


if __name__ == "__main__":
    # Update these paths according to your project structure.
    train_csv = "data/dataset/v2/training_set.csv"
    test_csv = "data/dataset/v2/test_set.csv"
    xgb_model_path = "models/v1_xgb_model.pkl"
    llm_config_path = "config/llm.yaml"
    faiss_db_dir = "data/vector_db/v1"

    results = compare_models(test_csv, train_csv, xgb_model_path, llm_config_path, faiss_db_dir, sample_size=1000,
                             k_retrieval=5)
    print("Comparison Metrics for first 1000 test instances:")
    print(results)
