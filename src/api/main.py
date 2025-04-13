from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import faiss
import pickle

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.models.xgboost_model import XGBFraudModel
from src.models.llm_model import LLMFraudModel
from src.utils.helpers import load_faiss_db
from src.preprocessing.xgb_preprocessor import XGBPreprocessor
from src.preprocessing.llm_preprocessor import LLMPreprocessor

app = FastAPI(title="Fraud Detection API")

DB_DIR = "data/vector_db/v1"
TRAINING_SET = 'data/dataset/v2/training_set.csv'


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


