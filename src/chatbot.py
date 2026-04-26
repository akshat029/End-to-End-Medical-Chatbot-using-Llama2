from __future__ import annotations

from typing import TypedDict

from src.llm_client import LLMClient
from src.vector_store import LocalVectorStore


class ChatbotResponse(TypedDict):
    answer: str
    sources: list[str]


class MedicalChatbot:
    def __init__(self):
        self.vector_store = LocalVectorStore()
        self.vector_store.load()
        self.llm = LLMClient()

    def ask(self, question: str) -> ChatbotResponse:
        hits = self.vector_store.search(question)
        if not hits:
            return {
                "answer": "I could not find relevant medical context in the indexed documents.",
                "sources": [],
            }

        context = "\n\n".join([f"Source: {h['source']}\n{h['text']}" for h in hits])
        answer = self.llm.generate(question=question, context=context)
        sources = sorted({h["source"] for h in hits})

        return {"answer": answer, "sources": sources}
