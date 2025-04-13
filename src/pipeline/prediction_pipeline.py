import time


class PredictionPipeline:
    def __init__(self, xgb_model, llm_model, default_model="llm"):
        """
        Initialize the pipeline with two models.

        Args:
            xgb_model: The XGBFraudModel instance.
            llm_model: The LLMFraudModel instance.
            default_model (str): "llm" or "xgb". Default is "llm".
        """
        self.xgb_model = xgb_model
        self.llm_model = llm_model
        self.default_model = default_model.lower()

    def predict(self, **kwargs):
        """
        Predict using the selected model.
        For LLM predictions, kwargs should include:
          - dataset_path: Path to training CSV (for global examples)
          - faiss_index: Pre-built FAISS index
          - metadata: FAISS metadata (list of dictionaries)
          - k_retrieval: Number of retrieval examples (default 5)
        For XGB predictions, kwargs should include:
          - dataframe: A preprocessed DataFrame.
        Returns a dict with the prediction and latency.
        """
        start = time.time()
        if self.default_model == "llm":
            pred = self.llm_model.full_predict(
                kwargs.get("data"),
                kwargs.get("dataset_path"),
                kwargs.get("faiss_index"),
                kwargs.get("metadata"),
                k_retrieval=kwargs.get("k_retrieval", 5)
            )
        elif self.default_model == "xgb":
            pred = self.xgb_model.predict(kwargs.get("dataframe"))[0]
        else:
            raise ValueError("Invalid model selection; must be 'llm' or 'xgb'.")
        latency = time.time() - start
        return {"prediction": int(pred), "latency": latency}
