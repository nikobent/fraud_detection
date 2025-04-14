FROM python:3.11-slim

WORKDIR /app

# System-level dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the entire project code
COPY ./src ./src
COPY ./download_models ./download_models
COPY ./data ./data
COPY ./config ./config

RUN mkdir -p /download_models

# Download embedding model and LLM model into local folders
RUN python download_models/download_all_mini.py
RUN python download_models/download_falcon.py

# Expose the API port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
