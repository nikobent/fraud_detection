# Fraud Detection Application

This project implements a fraud detection solution for FRISS Insurances. The application uses machine learning to detect fraudulent claims by combining two approaches: a modern LLM-based method with retrieval-augmented generation (RAG) and (optionally) a traditional XGBoost model (not used in the current Docker configuration).

In this version, the focus is on the LLM-based approach. The system loads a transformer model locally and uses prompt engineering (with support for multiple strategies) along with a FAISS index for retrieving similar historical examples. A REST API is provided via FastAPI with endpoints for scoring individual claims and evaluating the prompt strategies.

## Prerequisites

- Python 3.11+
- Docker (optional, but recommended for easy deployment)
- Git (to clone the repository)

---

## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/fraud-detection-api.git
   cd fraud-detection-api

   ```bash
        Copy
        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
        pip install --upgrade pip
        pip install -r requirements.txt 
   
Configuration:

Edit config/llm.yaml as needed to define your LLM parameters (model_name, prompt_prefix, etc.).

Place your training and test CSV files in data/dataset/v2/.

Ensure your FAISS index (and metadata) are available in data/vector_db/v1/.

Running the Application
Using Docker
Build the Docker Image:

From the project root, run:

       ```bash
    docker build -t fraud-detector .

Run the Docker Container:

To bind the container's port 8000 to your host's port 8000, run:

   ```bash
        docker run -p 8000:8000 fraud-detector
        
 Your API will be accessible at http://localhost:8000.

Running Locally with FastAPI
Activate Your Virtual Environment:

   ```bash
        source venv/bin/activate
        Start the FastAPI App:

From the project root, run:

   ```bash

        uvicorn src.api.main:app --reload
Access the API:

Swagger UI: http://127.0.0.1:8000/docs

Redoc: http://127.0.0.1:8000/redoc

API Endpoints
/score Endpoint
Method: POST

Input: A JSON payload containing all required claim fields (see PredictionRequest). Default values are provided to ease testing.

Output: A JSON response containing the prediction (0 for non-fraud, 1 for fraud) and the latency.

/evaluate_prompt Endpoint
Method: POST

Functionality:
Evaluates two different prompt strategies for the LLM-based model:

"similar": Uses 5 randomly selected global examples.

"fraud": Uses 5 fraudulent examples.

It processes a subset of the test set (default 500 instances), computing per-class and macro-averaged F1-score, precision, recall, as well as total and average latency for each strategy.

Output: A JSON object with evaluation metrics for each prompt strategy.
