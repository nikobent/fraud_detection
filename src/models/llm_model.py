import os
import yaml
from .base_model import FraudModel

from langchain.prompts import (
    SystemMessagePromptTemplate, HumanMessagePromptTemplate
)
from langchain.chains import LLMChain

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Import the LLM-specific preprocessor
from src.preprocessing.llm_preprocessor import LLMPreprocessor

# For local model loading using transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


class LLMFraudModel(FraudModel):
    def __init__(self, config: dict):
        """
        Initialize the LLMFraudModel using parameters from the configuration.
        This version loads the model locally from a specified directory if present;
        otherwise, it downloads it using the model name.
        """
        self.config = config
        self.model_name = config.get("model_name", "google/flan-t5-base")
        self.temperature = config.get("temperature", 0.7)
        self.max_new_tokens = config.get("max_new_tokens", 150)
        self.prompt_prefix = config.get("prompt_prefix", "")
        # Now we use 5 examples (global and retrieval) instead of 10.
        self.num_retrievals = config.get("num_retrievals", 5)

        # Define the local model directory.
        local_model_dir = "models/flan-t5-base"
        try:
            if os.path.exists(local_model_dir) and os.path.isdir(local_model_dir):
                tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
                model = AutoModelForSeq2SeqLM.from_pretrained(local_model_dir)
            else:
                print(f"Local model directory '{local_model_dir}' not found; downloading '{self.model_name}'...")
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            # Create a text2text-generation pipeline for local inference.
            self.llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        except Exception as e:
            raise Exception(f"Failed to load local model: {str(e)}")

        # Use the prompt_prefix as the system instruction.
        self.system_instruction = self.prompt_prefix

        try:
            # Build a simple prompt template.
            self.prompt_template = SystemMessagePromptTemplate.from_template(self.prompt_prefix)
        except Exception as e:
            raise Exception(f"Failed to construct prompt template: {str(e)}")

        # We’ll build our complete prompt in our own build_prompt method.

    @classmethod
    def from_config(cls, config_path: str = "config/llm.yaml"):
        """
        Loads configuration from a YAML file and instantiates LLMFraudModel.
        """
        full_path = os.path.join(os.getcwd(), config_path)
        try:
            with open(full_path, "r") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"Failed to load config from {config_path}: {str(e)}")
        return cls(config)

    def build_prompt(self, query: str, global_examples: list, retrieved_examples: list) -> str:
        """
        Build the final prompt by merging:
          - Global examples,
          - Retrieved examples, and
          - The new claim.
        """
        try:
            global_text = "\n".join([f"- {ex}" for ex in global_examples])
            retrieved_text = "\n".join([f"- {ex}" for ex in retrieved_examples])
            """
            "Global Examples, 1 is fraud and 4 are non-fraud but that doesn't showcase the actual distirbution of classes\n"
            f"{global_text}\n\n"""
            prompt = (
                f"{self.system_instruction}\n\n"
                "The 5 most similar claims from our database:\n"
                f"{retrieved_text}\n\n"
                f"New Claim: {query}\n\n"
                "Based on the above, is this claim fraudulent? Answer 'Yes' or 'No' and provide a brief explanation."
            )
            return prompt
        except Exception as e:
            raise Exception(f"Failed to build prompt: {str(e)}")

    def get_global_examples(self, dataset_path: str, total_examples: int = 5) -> list:
        """
        Load the dataset and sample a total of 5 global examples.
        Each example is converted to a natural language narrative with its label appended.
        """
        try:
            df = pd.read_csv(dataset_path)
            if len(df) < total_examples:
                sample_df = df
            else:
                sample_df = df.sample(n=total_examples)
            preproc = LLMPreprocessor()
            examples = []
            for _, row in sample_df.iterrows():
                row_dict = row.to_dict()
                transformed = preproc.transform(row_dict)
                sentence = preproc.to_sentence(transformed)
                # Append the label for clarity.
                label = row.get("label", 0)
                label_str = "Fraud" if int(label) == 1 else "Non-Fraud"
                examples.append(f"{sentence} (Label: {label_str})")
            return examples
        except Exception as e:
            raise Exception(f"Failed to get global examples: {str(e)}")

    def retrieve_similar_examples(self, query: str, faiss_index, metadata: list, k: int = 5) -> list:
        """
        Retrieve the k most similar historical claim narratives from a pre-built FAISS index.
        Appends the label from metadata to each retrieved example.
        """
        try:
            embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            query_embedding = embed_model.encode([query])
            distances, indices = faiss_index.search(np.array(query_embedding), k)
            retrieved = []
            for i in indices[0]:
                meta = metadata[i]
                text = meta.get("text", "")
                label = meta.get("label", 0)
                label_str = "Fraud" if int(label) == 1 else "Non-Fraud"
                retrieved.append(f"{text} (Label: {label_str})")
            return retrieved
        except Exception as e:
            raise Exception(f"Failed to retrieve similar examples: {str(e)}")

    def full_predict(self, data: dict, dataset_path: str, faiss_index, metadata: list, k_retrieval: int = 5) -> int:
        """
        Execute the full prediction chain:
          - Sample 5 global examples (with label included) from the dataset,
          - Retrieve 5 similar examples using the FAISS index,
          - Build the final prompt,
          - Call the local inference pipeline,
          - Return 1 if the generated response indicates fraud (if "yes" is found), else 0.
        """
        try:
            query = data.get("query", "")
            global_examples = self.get_global_examples(dataset_path, total_examples=5)
            retrieved_examples = self.retrieve_similar_examples(query, faiss_index, metadata, k=k_retrieval)
            final_prompt = self.build_prompt(query, global_examples, retrieved_examples)
            print("Final Prompt:\n", final_prompt)
        except Exception as e:
            raise Exception(f"Error during prompt assembly: {str(e)}")

        try:
            result = self.llm(final_prompt)
            generated_text = result[0]['generated_text']
            # Return 1 if the response includes "yes", else 0.
            return 1 if "yes" in generated_text.lower() else 0
        except Exception as e:
            raise Exception(f"LLM inference failed: {str(e)}")

    def predict(self, data: dict) -> int:
        raise NotImplementedError("Use full_predict with dataset and FAISS inputs for complete prediction.")

    def fit(self, X, y):
        pass  # Not applicable for prompt-based inference.

    def save(self, path: str):
        pass  # Not applicable; configuration-based.

    @classmethod
    def load(cls, path: str):
        return cls.from_config()
