from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import faiss
import pickle
import numpy as np
import time

from sklearn.metrics import f1_score, precision_score, recall_score

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.models.xgboost_model import XGBFraudModel
from src.models.llm_model import LLMFraudModel
from src.utils.helpers import load_faiss_db
from src.preprocessing.xgb_preprocessor import XGBPreprocessor
from src.preprocessing.llm_preprocessor import LLMPreprocessor

app = FastAPI(title="Fraud Detection API")

DB_DIR = "data/vector_db/v1"
TRAINING_SET = 'data/dataset/v2/training_set.csv'
TEST_SET = 'data/dataset/v2/test_set.csv'


class PredictionRequest(BaseModel):
    model: str = "llm"  # Allowed values: "xgb" or "llm"; default is "llm"
    sys_dataspecification_version: str = "4.5"
    sys_claimid: str = "MTR-338957796-02"
    claim_amount_claimed_total: float = 2433.0
    claim_causetype: str = "Collision"
    claim_date_occurred: str = "20121022"
    claim_date_reported: str = "20121127"
    claim_location_urban_area: int = 1
    object_make: str = "VOLKSWAGEN"
    object_year_construction: int = 2008
    policy_fleet_flag: int = 0
    policy_profitability: str = "Low"

@app.post("/score")
async def score(request: PredictionRequest):
    # convert the input data into dic
    input_data = request.dict()
    model_type = input_data.pop("model").lower()

    if model_type == "xgb":
        # init model class
        try:
            print("yo1")
            xgb_model = XGBFraudModel.load()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"XGB model loading error: {e}")

        # preprocess input and get prediction
        try:
            print("yo2")
            preproc = XGBPreprocessor()
            print("yo3")
            df = preproc.transform(input_data)
            print("yo4")
            pipeline_instance = PredictionPipeline(xgb_model=xgb_model, llm_model=None, default_model="xgb")
            print("yo5")
            result = pipeline_instance.predict(data=None, dataframe=df)
            print("yo6")
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    elif model_type == "llm":
        try:
            print("yo2")
            llm_model = LLMFraudModel.from_config()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM model loading error: {e}")

        # Load FAISS DB only if LLM branch is used.
        try:
            print("yo3")
            faiss_index, metadata = load_faiss_db(DB_DIR)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"FAISS DB loading error: {e}")

        try:
            # Use the LLM preprocessor to convert the full input into a narrative query.
            print("yo4")
            preproc = LLMPreprocessor()
            print("yo5")
            transformed = preproc.transform(input_data)
            print("yo6")
            query = preproc.to_sentence(transformed)
            data = {"query": query}
            print("yo7")
            pipeline_instance = PredictionPipeline(xgb_model=None, llm_model=llm_model, default_model="llm")
            print("yo8")
            result = pipeline_instance.predict(
                data=data,
                dataset_path=TRAINING_SET,
                faiss_index=faiss_index,
                metadata=metadata,
                k_retrieval=5
            )
            print("yo9")
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail="Invalid model specified.")


@app.post("/evaluate")
async def evaluate(n: int = 500):
    """
    Evaluate both models on the first n test instances (default: 500).
    For each model, compute per-class and average F1-score, precision, and recall,
    and record the total and average latency.
    Prints a status update every 50 instances.

    If any error occurs during evaluation, the exception will be raised.
    """
    # Load the test dataset.
    try:
        test_df = pd.read_csv(TEST_SET)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading test CSV: {e}")

    if len(test_df) > n:
        test_df = test_df.iloc[:n]
    true_labels = test_df["sys_fraud"].values  # adjust if your label column is named differently

    # Initialize accumulators.
    xgb_preds, llm_preds = [], []
    xgb_latencies, llm_latencies = [], []

    # Create one instance of the LLM preprocessor.
    preproc = LLMPreprocessor()

    # Loop over each test instance (if any error occurs here, it will propagate)
    for i, (_, row) in enumerate(test_df.iterrows()):
        data = row.to_dict()
        """
        # ----- XGB Prediction -----
        # Load the model and predict (will raise on error)
        xgb_model = XGBFraudModel.load()
        start_time = time.time()
        xgb_input = pd.DataFrame([data])
        xgb_pred = xgb_model.predict(xgb_input)[0]
        xgb_latency = time.time() - start_time
        xgb_preds.append(xgb_pred)
        xgb_latencies.append(xgb_latency)
        """
        # ----- LLM Prediction -----
        llm_model = LLMFraudModel.from_config()
        transformed = preproc.transform(data)
        query = preproc.to_sentence(transformed)
        start_time = time.time()
        input_data = {"query": query}
        faiss_index, metadata = load_faiss_db(DB_DIR)
        pipeline_instance = PredictionPipeline(xgb_model=None, llm_model=llm_model, default_model="llm")
        result = pipeline_instance.predict(
            data=input_data,
            dataset_path=TRAINING_SET,
            faiss_index=faiss_index,
            metadata=metadata,
            k_retrieval=5
        )
        llm_latency = time.time() - start_time
        llm_preds.append(result["prediction"])
        llm_latencies.append(llm_latency)

        # Print a status update every 50 instances.
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1} out of {n} instances")

    """
    # Compute metrics for XGB.
    xgb_f1 = f1_score(true_labels, xgb_preds, average=None)
    xgb_precision = precision_score(true_labels, xgb_preds, average=None, zero_division=0)
    xgb_recall = recall_score(true_labels, xgb_preds, average=None, zero_division=0)
    xgb_metrics = {
        "f1_score_per_class": xgb_f1.tolist(),
        "precision_per_class": xgb_precision.tolist(),
        "recall_per_class": xgb_recall.tolist(),
        "avg_f1_score": f1_score(true_labels, xgb_preds, average="macro"),
        "avg_precision": precision_score(true_labels, xgb_preds, average="macro", zero_division=0),
        "avg_recall": recall_score(true_labels, xgb_preds, average="macro", zero_division=0),
        "total_latency": sum(xgb_latencies),
        "avg_latency": np.mean(xgb_latencies)
    }
    """
    # Compute metrics for LLM.
    llm_f1 = f1_score(true_labels, llm_preds, average=None)
    llm_precision = precision_score(true_labels, llm_preds, average=None, zero_division=0)
    llm_recall = recall_score(true_labels, llm_preds, average=None, zero_division=0)
    llm_metrics = {
        "f1_score_per_class": llm_f1.tolist(),
        "precision_per_class": llm_precision.tolist(),
        "recall_per_class": llm_recall.tolist(),
        "avg_f1_score": f1_score(true_labels, llm_preds, average="macro"),
        "avg_precision": precision_score(true_labels, llm_preds, average="macro", zero_division=0),
        "avg_recall": recall_score(true_labels, llm_preds, average="macro", zero_division=0),
        "total_latency": sum(llm_latencies),
        "avg_latency": np.mean(llm_latencies)
    }

    metrics = { "llm": llm_metrics}#"xgb": xgb_metrics,
    print(metrics)
    return metrics