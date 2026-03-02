import argparse
import json
import os
import sys
import time

import config
config.validate()

from parser import parse_repo, build_graph, save_graph
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone as PineconeClient


def _symbol_to_document(sym, repo_name: str) -> Document:
    parent_note = f" inside class {sym.parent_class}" if sym.parent_class else ""
    page_content = (
        f"File: {sym.file_path}\n"
        f"Type: {sym.kind} named {sym.name}{parent_note}\n"
        f"Signature: {sym.signature or 'N/A'}\n"
        f"Description: {sym.docstring or 'None'}\n"
        f"Calls: {', '.join(sym.calls) if sym.calls else 'nothing'}\n"
        f"\n{sym.source_code}"
    )
    metadata = {
        "chunk_id":     sym.id,
        "file_path":    sym.file_path,
        "name":         sym.name,
        "kind":         sym.kind,
        "line_start":   sym.line_start,
        "line_end":     sym.line_end,
        "parent_class": sym.parent_class,
        "source_code":  sym.source_code,
        "repo_name":    repo_name,
    }
    return Document(page_content=page_content, metadata=metadata)


def ingest_repo(repo_path: str, embeddings, pc, session_results: list):
    if not os.path.isdir(repo_path):
        print(f"ERROR: Not a directory: {repo_path}")
        sys.exit(1)

    repo_name = os.path.basename(os.path.normpath(repo_path))
    print(f"\nProcessing {repo_name}...")

    symbols = parse_repo(repo_path)
    print(f"  Found {len(symbols)} symbols")

    graph = build_graph(symbols)
    os.makedirs(config.DATA_DIR, exist_ok=True)
    graph_path = os.path.join(config.DATA_DIR, f"{repo_name}_graph.json")
    save_graph(graph, graph_path)
    print(f"  Graph saved: {len(graph['nodes'])} nodes, "
          f"{sum(len(v) for v in graph['edges'].values())} edges")

    documents = [_symbol_to_document(s, repo_name) for s in symbols]

    pinecone_index = pc.Index(config.PINECONE_INDEX_NAME)
    vector_store = PineconeVectorStore(
        index=pinecone_index,
        embedding=embeddings,
        namespace=repo_name,
    )
    # Upload in batches to avoid rate limits
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        vector_store.add_documents(documents[i:i + batch_size])
    print(f"  Uploaded {len(documents)} chunks to Pinecone (namespace: {repo_name})")

    session_results.append({
        "repo_name":   repo_name,
        "repo_root":   os.path.abspath(repo_path),
        "chunk_count": len(documents),
    })


def main():
    ap = argparse.ArgumentParser(
        description="Ingest two local repos into Pinecone for repo-chatbot."
    )
    ap.add_argument("repo1_path", help="Path to the first repository")
    ap.add_argument("repo2_path", help="Path to the second repository")
    args = ap.parse_args()

    start = time.time()

    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
    )

    pc = PineconeClient(api_key=config.PINECONE_API_KEY)

    session_results = []
    ingest_repo(args.repo1_path, embeddings, pc, session_results)
    ingest_repo(args.repo2_path, embeddings, pc, session_results)

    os.makedirs(os.path.dirname(config.SESSION_FILE), exist_ok=True)
    with open(config.SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump({"repos": session_results}, f, indent=2)
    print(f"\nSession saved to {config.SESSION_FILE}")
    print(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
