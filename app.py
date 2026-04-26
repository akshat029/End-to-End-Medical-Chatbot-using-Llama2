from __future__ import annotations

import streamlit as st

from src.chatbot import MedicalChatbot


st.set_page_config(page_title="Medical Chatbot (Llama2)", page_icon="🩺", layout="centered")
st.title("🩺 End-to-End Medical Chatbot using Llama2")

if "chatbot" not in st.session_state:
    try:
        st.session_state.chatbot = MedicalChatbot()
    except Exception as exc:
        st.error(
            "Initialization failed. Ensure you ran `python ingest.py`, configured `.env`, and set HF token/model access."
        )
        st.exception(exc)
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a medical question based on your indexed documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.chatbot.ask(question)
            answer = result["answer"]
            sources = result["sources"]

        st.markdown(answer)
        if sources:
            st.caption("Sources: " + " | ".join(sources))

    st.session_state.messages.append({"role": "assistant", "content": answer})
