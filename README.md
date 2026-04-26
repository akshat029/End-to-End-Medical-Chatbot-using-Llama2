# End-to-End Medical Chatbot using Llama2

This repository now contains a complete retrieval-augmented medical chatbot pipeline:

- Document ingestion (`.txt`, `.md`, `.pdf`)
- Chunking + embeddings
- Local FAISS vector index
- Llama2 response generation through Hugging Face Inference API
- Streamlit chat UI

## Project Structure

```text
.
├── app.py
├── ingest.py
├── requirements.txt
├── data/
├── vector_store/
└── src/
    ├── chatbot.py
    ├── config.py
    ├── data_loader.py
    ├── llm_client.py
    └── vector_store.py
```

## Requirements

- Python 3.10+
- Hugging Face account with access to `meta-llama/Llama-2-7b-chat-hf`
- Hugging Face API token

## Setup

1. Create virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

2. Configure environment variables:

```bash
cp .env.example .env
```

Set at least:

- `HUGGINGFACEHUB_API_TOKEN`
- (Optional) `LLM_MODEL_ID`

3. Add your medical documents into `data/`.

## Build the Vector Index

```bash
python ingest.py
```

## Run the Chatbot

```bash
streamlit run app.py
```

## APIs / Inputs required from you

To run this project end-to-end, I require:

1. **Hugging Face token** (`HUGGINGFACEHUB_API_TOKEN`)
2. **Approved access to Llama2 model** on Hugging Face (gated model access)
3. **Your medical source documents** inside `data/`

Without these, ingestion and app startup can still run partially, but answer generation with Llama2 will fail.
