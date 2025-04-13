import time

class PredictionPipeline:
    def __init__(self, xgb_model, llm_model, default_model="llm"):
        self.xgb_model = xgb_model
        self.llm_model = llm_model
        self.default_model = default_model  # Options: "llm" or "xgb"

    def predict(self, data: dict, **kwargs):
        """
        Predict using the selected model.
        kwargs should include:
          - dataset_path: Path to training CSV for global examples.
          - faiss_index: The pre-built FAISS index.
          - metadata: Metadata corresponding to the FAISS index.
          - k_retrieval: Number of retrieval examples to use (default 5).
        """
        start = time.time()
        if self.default_model.lower() == "llm":
            pred = self.llm_model.full_predict(
                data,
                kwargs.get("dataset_path"),
                kwargs.get("faiss_index"),
                kwargs.get("metadata"),
                k_retrieval=kwargs.get("k_retrieval", 5)
            )
        elif self.default_model.lower() == "xgb":
            # For demo purposes, we assume xgb_model.predict returns a list/array and that
            # input data is preprocessed into a DataFrame.
            pred = self.xgb_model.predict(kwargs.get("dataframe"))[0]
        else:
            raise ValueError("Invalid model selection.")
        latency = time.time() - start
        return {"prediction": int(pred), "latency": latency}
