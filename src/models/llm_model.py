import os
import yaml
from .base_model import FraudModel

# We'll use LangChain's prompt templates for structured formatting.
from langchain.prompts import SystemMessagePromptTemplate
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Import our LLM-specific preprocessor.
from src.preprocessing.llm_preprocessor import LLMPreprocessor

# For local model loading using transformers.
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


class LLMFraudModel(FraudModel):
    def __init__(self, config: dict, prompt_type: str == "similar"):
        """
        Initialize the LLMFraudModel using parameters from the configuration.
        This class loads the model locally from a directory (or downloads if not found)
        based on the model name specified in the config.
        """
        self.config = config
        self.model_name = config.get("model_name", "google/flan-t5-base")
        self.temperature = config.get("temperature", 0.7)
        self.max_new_tokens = config.get("max_new_tokens", 150)
        self.prompt_prefix = config.get("prompt_prefix", "")
        self.num_retrievals = config.get("num_retrievals", 5)
        self.prompt_type = prompt_type

        # Determine local model directory from the model name (e.g., "google/flan-t5-base" → "models/flan-t5-base")
        local_model_dir = os.path.join("models", "falcon_llm")
        try:
            if os.path.exists(local_model_dir) and os.path.isdir(local_model_dir):
                tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
                model = AutoModelForSeq2SeqLM.from_pretrained(local_model_dir)
            else:
                print(f"Local model directory '{local_model_dir}' not found; downloading '{self.model_name}'...")
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        except Exception as e:
            raise Exception(f"Failed to load local model: {str(e)}")

        self.system_instruction = self.prompt_prefix
        try:
            self.prompt_template = SystemMessagePromptTemplate.from_template(self.prompt_prefix)
        except Exception as e:
            raise Exception(f"Failed to construct prompt template: {str(e)}")

    @classmethod
    def from_config(cls, config_path: str = "config/llm.yaml", prompt_type: str = "similar"):
        """
        Loads configuration from a YAML file and instantiates the LLMFraudModel.

        Args:
            config_path (str): Path to the configuration file.
            prompt_type (str): Strategy to use ("similar" or "fraud").
        """
        full_path = os.path.join(os.getcwd(), config_path)
        try:
            with open(full_path, "r") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"Failed to load config from {config_path}: {str(e)}")
        return cls(config, prompt_type=prompt_type)

    def build_prompt(self, query: str, global_examples: list, retrieved_examples: list) -> str:
        """
        Build the final prompt by merging:
          - Global examples (with labels),
          - Retrieved examples (with labels), and
          - The new claim (query).
        """
        try:
            global_text = "\n".join([f"- {ex}" for ex in global_examples])
            retrieved_text = "\n".join([f"- {ex}" for ex in retrieved_examples])
            prompt = (
                f"{self.system_instruction}\n\n"
                "Global Examples (5 examples with labels):\n"
                f"{global_text}\n\n"
                "Retrieved Examples (5 most similar historical claims with labels):\n"
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

        If prompt_type is "fraud", sample 5 examples where label == 1.
        Otherwise, sample 5 random examples (regardless of label).
        Each example is converted to a narrative using LLMPreprocessor and appended with its label.
        """
        try:
            df = pd.read_csv(dataset_path)
            if self.prompt_type == "fraud":
                fraud_df = df[df["label"] == 1]
                # If there are insufficient fraud examples, use all available.
                sample_df = fraud_df.sample(n=total_examples) if len(fraud_df) >= total_examples else fraud_df
                strategy = "fraud"
            else:
                # "similar" or default: random sampling.
                sample_df = df.sample(n=total_examples) if len(df) >= total_examples else df
                strategy = "similar"
            preproc = LLMPreprocessor()
            examples = []
            for _, row in sample_df.iterrows():
                row_dict = row.to_dict()
                transformed = preproc.transform(row_dict)
                sentence = preproc.to_sentence(transformed)
                label = row.get("label", 0)
                label_str = "Fraud" if int(label) == 1 else "Non-Fraud"
                examples.append(f"{sentence} (Label: {label_str})")
            print(f"Using {strategy} strategy for global examples.")
            return examples
        except Exception as e:
            raise Exception(f"Failed to get global examples: {str(e)}")

    def retrieve_similar_examples(self, query: str, faiss_index, metadata: list, k: int = 5) -> list:
        """
        Retrieve the top k most similar historical claim narratives from a pre-built FAISS index.
        Append the label from metadata to each retrieved example.
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
          1. Sample 5 global examples (with labels) from the training dataset.
          2. Retrieve 5 similar historical examples from the FAISS index.
          3. Build the final prompt.
          4. Execute local inference using our pipeline.
          5. Return 1 if the generated response indicates fraud (i.e. contains "yes"), else 0.
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
