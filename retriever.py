import os

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient

from parser import load_graph, get_neighbors
import config


class Retriever:
    def __init__(self, namespace):
        """
        namespace: str (one repo) or list[str] (both repos).
        """
        self.namespaces = [namespace] if isinstance(namespace, str) else namespace

        embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
        )

        self._pc = PineconeClient(api_key=config.PINECONE_API_KEY)
        self._index = self._pc.Index(config.PINECONE_INDEX_NAME)

        self.stores: dict = {}
        for ns in self.namespaces:
            self.stores[ns] = PineconeVectorStore(
                index=self._index,
                embedding=embeddings,
                namespace=ns,
            )

        self.graphs: dict = {}
        for ns in self.namespaces:
            graph_path = os.path.join(config.DATA_DIR, f"{ns}_graph.json")
            if os.path.exists(graph_path):
                self.graphs[ns] = load_graph(graph_path)
            else:
                self.graphs[ns] = {"nodes": {}, "edges": {}}

    def retrieve(self, query: str) -> list:
        # ── Step 1: similarity search ────────────────────────────────────────
        raw_results = []
        for ns in self.namespaces:
            pairs = self.stores[ns].similarity_search_with_score(query, k=config.TOP_K)
            for doc, score in pairs:
                raw_results.append((score, doc, ns))

        # Merge and keep top TOP_K overall (sorted by score desc)
        raw_results.sort(key=lambda x: x[0], reverse=True)
        seeds = raw_results[:config.TOP_K]

        # ── Step 2: graph expansion ──────────────────────────────────────────
        chunks_by_id: dict = {}

        for score, doc, ns in seeds:
            meta = doc.metadata
            chunk_id = meta.get("chunk_id", "")
            chunks_by_id[chunk_id] = {
                "chunk_id":       chunk_id,
                "source_code":    meta.get("source_code", doc.page_content),
                "file_path":      meta.get("file_path", ""),
                "name":           meta.get("name", ""),
                "kind":           meta.get("kind", ""),
                "line_start":     meta.get("line_start", 0),
                "line_end":       meta.get("line_end", 0),
                "repo_name":      meta.get("repo_name", ns),
                "retrieval_type": "direct",
            }

            # expand via graph
            graph = self.graphs.get(ns, {})
            neighbor_ids = get_neighbors(
                graph, chunk_id,
                depth=config.GRAPH_DEPTH,
                max_per_node=config.MAX_NEIGHBORS,
            )

            if neighbor_ids:
                try:
                    fetch_resp = self._index.fetch(ids=neighbor_ids, namespace=ns)
                    fetched = fetch_resp.get("vectors", {}) if isinstance(fetch_resp, dict) \
                              else getattr(fetch_resp, "vectors", {})
                    for nid, vec in fetched.items():
                        if nid not in chunks_by_id:
                            nmeta = vec.get("metadata", {}) if isinstance(vec, dict) \
                                    else getattr(vec, "metadata", {})
                            chunks_by_id[nid] = {
                                "chunk_id":       nid,
                                "source_code":    nmeta.get("source_code", ""),
                                "file_path":      nmeta.get("file_path", ""),
                                "name":           nmeta.get("name", ""),
                                "kind":           nmeta.get("kind", ""),
                                "line_start":     nmeta.get("line_start", 0),
                                "line_end":       nmeta.get("line_end", 0),
                                "repo_name":      nmeta.get("repo_name", ns),
                                "retrieval_type": "graph_neighbor",
                            }
                except Exception:
                    pass  # graph expansion is best-effort

        return list(chunks_by_id.values())


def build_context(chunks: list, max_chunks: int) -> str:
    """Format retrieved chunks into a numbered context block for the LLM."""
    direct   = [c for c in chunks if c["retrieval_type"] == "direct"]
    neighbor = [c for c in chunks if c["retrieval_type"] == "graph_neighbor"]

    selected = direct[:max_chunks]
    remaining = max_chunks - len(selected)
    if remaining > 0:
        selected += neighbor[:remaining]

    selected.sort(key=lambda c: (c["repo_name"], c["file_path"], c["line_start"]))

    parts = ["=== CODEBASE CONTEXT ===\n"]
    for i, c in enumerate(selected, 1):
        header = (
            f"[{i}] [{c['repo_name']}] {c['file_path']} | "
            f"{c['kind']}: {c['name']} "
            f"(lines {c['line_start']}-{c['line_end']})"
        )
        parts.append(header)
        parts.append("---")
        parts.append(c["source_code"])
        parts.append("")

    return "\n".join(parts)
