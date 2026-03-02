import json
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from retriever import Retriever, build_context
import config

_SYSTEM_PROMPT = (
    "You are a helpful software engineer assistant. "
    "Answer questions about the codebase using only the source code provided below as context. "
    "Always mention the file path and function name when referencing code. "
    "If the context does not contain the answer, say so — do not guess. "
    "Use markdown code blocks for any code snippets."
)

_REWRITE_SYSTEM = (
    "Rewrite the user's question to be fully self-contained "
    "using the conversation history. Output only the rewritten question, nothing else."
)


class Responder:
    def __init__(self):
        self.llm      = ChatOpenAI(model=config.LLM_MODEL, temperature=0.2,
                                   openai_api_key=config.OPENAI_API_KEY)
        self.rewriter = ChatOpenAI(model=config.LLM_MODEL, temperature=0,
                                   openai_api_key=config.OPENAI_API_KEY)
        self.history: list = []
        self.last_sources: list = []

        if not os.path.exists(config.SESSION_FILE):
            print("WARNING: session.json not found. Run ingest.py first.")
            self.repos = []
        else:
            with open(config.SESSION_FILE, "r", encoding="utf-8") as f:
                self.repos = json.load(f).get("repos", [])

    def _rewrite_question(self, question: str) -> str:
        if not self.history:
            return question
        # Use last 4 turns (8 messages)
        recent = self.history[-8:]
        messages = [SystemMessage(content=_REWRITE_SYSTEM)]
        messages += recent
        messages.append(HumanMessage(content=question))
        response = self.rewriter.invoke(messages)
        return response.content.strip()

    def chat(self, question: str, namespace) -> dict:
        rewritten = self._rewrite_question(question)

        retriever = Retriever(namespace)
        chunks    = retriever.retrieve(rewritten)
        context   = build_context(chunks, config.MAX_CHUNKS)
        self.last_sources = chunks

        messages = [SystemMessage(content=_SYSTEM_PROMPT)]
        messages += self.history
        messages.append(HumanMessage(
            content=f"Context:\n{context}\n\nQuestion: {question}"
        ))

        response = self.llm.invoke(messages)
        answer   = response.content

        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=answer))

        return {
            "answer":          answer,
            "sources":         chunks,
            "rewritten_query": rewritten,
        }

    def clear(self):
        self.history = []
        self.last_sources = []
