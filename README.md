# Repo Chatbot

A chatbot that answers questions about two code repositories. It understands code structure using tree-sitter to map out functions and classes, then finds relevant code using both semantic search and a relationship graph before answering with GPT-5-mini.

## Prerequisites

- Python 3.10+
- OpenAI API key
- Pinecone account 

## Pinecone setup

Create a serverless index with:
- **Name:** gitchatbot
- **Dimensions:** 1536
- **Metric:** cosine

## Setup

1. Activate virtual env: source .venv/bin/activate
2. Ingest data: python ingest.py "../CTI-API" "../Service-Portal-API" 
3. Run app: python app.py
