from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_dir: str = os.getenv("DATA_DIR", "./data")
    vector_store_dir: str = os.getenv("VECTOR_STORE_DIR", "./vector_store")
    llm_model_id: str = os.getenv("LLM_MODEL_ID", "meta-llama/Llama-2-7b-chat-hf")
    hf_token: str | None = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    top_k: int = int(os.getenv("TOP_K", "4"))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "512"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))


settings = Settings()
