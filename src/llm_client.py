from __future__ import annotations

from huggingface_hub import InferenceClient

from src.config import settings


SYSTEM_PROMPT = (
    "You are a careful medical information assistant. "
    "You are not a doctor and do not provide diagnosis. "
    "Use only provided context, and say when context is insufficient."
)


class LLMClient:
    def __init__(self, model_id: str | None = None, token: str | None = None):
        self.model_id = model_id or settings.llm_model_id
        self.token = token or settings.hf_token
        if not self.token:
            raise ValueError(
                "HUGGINGFACEHUB_API_TOKEN is required for Llama2 inference. "
                "Set it in your .env file."
            )
        self.client = InferenceClient(model=self.model_id, token=self.token)

    def generate(self, question: str, context: str) -> str:
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer in clear concise language with a brief safety note."
        )

        response = self.client.text_generation(
            prompt=prompt,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            return_full_text=False,
        )
        return response.strip()
