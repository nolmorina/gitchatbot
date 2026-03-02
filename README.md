# Repo Chatbot

A chatbot that answers questions about two code repositories. It understands code structure using tree-sitter to map out functions and classes, then finds relevant code using both semantic search and a relationship graph before answering with GPT-4o-mini.

## Why it is better than simple search

Plain search finds files that match your keywords. This app also finds related functions — if you ask about authentication, it also pulls in the token validation and database lookup functions that authentication calls.

## Prerequisites

- Python 3.10+
- OpenAI API key
- Pinecone account (free tier works)

## Pinecone setup

Create a serverless index with:
- **Name:** repo-chatbot
- **Dimensions:** 1536
- **Metric:** cosine

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in OPENAI_API_KEY and PINECONE_API_KEY in .env
```

## Usage

```bash
# Step 1: ingest both repos (run once, or when repos change)
python ingest.py ./path/to/repo1 ./path/to/repo2

# Step 2: start the chat server
python app.py
```

Open **http://localhost:5000** in your browser.

## Architecture

```
SETUP (run once)
────────────────────────────────────────────────────
Your repo folders
     │
     ▼
parser.py  →  extracts functions, classes, methods
     │
     ├──→  builds + saves relationship graph to JSON
     │
     └──→  LangChain Documents (one per symbol)
                 │
                 ▼
          OpenAI Embeddings (text-embedding-3-small)
                 │
                 ▼
          Pinecone (one namespace per repo)

CHAT (every question)
────────────────────────────────────────────────────
Your question
     │
     ▼
responder.py → rewrites question (resolves "it", "that" etc.)
     │
     ▼
retriever.py → searches Pinecone for top matching chunks
     │
     ▼
retriever.py → expands via graph (pulls related functions)
     │
     ▼
retriever.py → formats context block
     │
     ▼
responder.py → GPT-4o-mini writes the answer
     │
     ▼
app.py → Flask → Browser UI (static/index.html)
```

## File reference

| File | Purpose |
|------|---------|
| `config.py` | All settings and environment variables |
| `parser.py` | Code parsing with tree-sitter + graph logic |
| `ingest.py` | Processes both repos and uploads to Pinecone |
| `retriever.py` | Finds relevant chunks via search + graph |
| `responder.py` | Manages conversation and calls the LLM |
| `app.py` | Flask web server and API routes |
| `static/index.html` | The entire chat interface |
