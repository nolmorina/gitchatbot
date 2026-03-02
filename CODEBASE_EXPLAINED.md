# Repo Chatbot — Complete Codebase Explanation

> **Who is this for?** This document is written for someone who has never written code before. By the end, you should understand exactly what this application does, why every design decision was made, and how you could build it yourself.

---

## Table of Contents

1. [What Does This App Do?](#1-what-does-this-app-do)
2. [The Big Picture — How It All Connects](#2-the-big-picture--how-it-all-connects)
3. [Core Concepts You Need to Know](#3-core-concepts-you-need-to-know)
4. [Python Basics Used in This Project](#4-python-basics-used-in-this-project)
5. [The Files — One by One](#5-the-files--one-by-one)
   - [`.env` — Secrets File](#51-env--secrets-file)
   - [`requirements.txt` — Dependency List](#52-requirementstxt--dependency-list)
   - [`config.py` — The Settings Dashboard](#53-configpy--the-settings-dashboard)
   - [`parser.py` — The Code Reader](#54-parserpy--the-code-reader)
   - [`ingest.py` — The Uploader](#55-ingestpy--the-uploader)
   - [`retriever.py` — The Smart Searcher](#56-retrieverpy--the-smart-searcher)
   - [`responder.py` — The Conversationalist](#57-responderpy--the-conversationalist)
   - [`app.py` — The Web Server](#58-apppy--the-web-server)
   - [`static/index.html` — The User Interface](#59-staticindexhtml--the-user-interface)
   - [`data/` — Saved State](#510-data--saved-state)
6. [Key Design Decisions Explained](#6-key-design-decisions-explained)
7. [The Full Journey of One Question](#7-the-full-journey-of-one-question)
8. [How to Build This Yourself — Step by Step](#8-how-to-build-this-yourself--step-by-step)

---

## 1. What Does This App Do?

Imagine you have two large software projects written by other engineers. You want to understand them — where is the login logic? How does the payment flow work? Which functions call each other?

Normally you would spend hours reading files. This app lets you **just ask questions in plain English** and get accurate answers with the exact file names and line numbers where the code lives.

**Example:**
> You type: *"How does authentication work?"*
> The app replies with a detailed explanation referencing `auth/middleware.js line 34` and `services/tokenService.go line 12`.

This is not a simple keyword search. The app actually **understands** the code structure — it knows which functions call which other functions, and when you ask about one thing, it automatically pulls in all the related pieces.

---

## 2. The Big Picture — How It All Connects

The application has two separate phases:

### Phase 1 — Setup (run once)

```
Your code repositories (folders of files)
              │
              ▼
        parser.py reads every file
        and extracts all functions, classes, methods
              │
              ├──► builds a "relationship map" (graph)
              │    showing which functions call which others
              │    → saved to data/CTI-API_graph.json
              │
              └──► turns each function/class into a "document"
                   → converts it to a vector (a list of numbers)
                     using OpenAI's embedding model
                   → uploads all vectors to Pinecone (a cloud database)
```

### Phase 2 — Chatting (every question)

```
You type a question
              │
              ▼
        responder.py rewrites the question
        (if you said "it" or "that", it figures out what you meant)
              │
              ▼
        retriever.py searches Pinecone for the most relevant
        code chunks (using vector similarity)
              │
              ▼
        retriever.py expands the search using the graph —
        "this function also calls these other functions, add them too"
              │
              ▼
        responder.py sends your question + all retrieved code
        to GPT-4o-mini, which writes the answer
              │
              ▼
        app.py (Flask web server) sends the answer back to your browser
              │
              ▼
        index.html displays it as a formatted chat bubble
```

---

## 3. Core Concepts You Need to Know

Before reading the code files, you need to understand five key ideas:

### 3.1 What is a Vector / Embedding?

Computers cannot understand text the way humans do. A technique called **embedding** converts text into a long list of numbers (a "vector"), where **similar meaning = similar numbers**.

For example, the phrase "user login" and "authentication function" will produce very similar vectors, even though they use completely different words. This is what allows the search to find relevant code even when you don't use the exact same words as the code.

OpenAI's `text-embedding-3-small` model converts any piece of text into a list of **1536 numbers**. This app converts every function and class in the repositories into such a list of numbers.

### 3.2 What is Pinecone?

Pinecone is a cloud database that specialises in storing and searching vectors. When you search, you give it a query (also converted to a vector), and it returns the stored vectors that are most mathematically similar — meaning the code chunks most semantically related to your question.

Think of it like a library that can return the books most relevant to your concept, not just your keywords.

### 3.3 What is a Graph (the relationship map)?

In this context, a **graph** is a data structure that records relationships. It has:
- **Nodes** — each node is one function, class, or code chunk
- **Edges** — each edge is a connection: "function A calls function B"

```
[authenticate] ──calls──► [validateToken]
[authenticate] ──calls──► [lookupUser]
[lookupUser]   ──calls──► [queryDatabase]
```

When the search finds `authenticate` is relevant, the graph expansion automatically also pulls in `validateToken`, `lookupUser`, and `queryDatabase` — giving GPT much richer context without you having to ask for each one.

**Why a graph?** Code is inherently relational. A function never exists in isolation — it always calls and is called by other functions. Plain keyword search ignores these relationships. A graph preserves them.

### 3.4 What is tree-sitter?

Tree-sitter is a tool that **parses source code** into a structured tree. Instead of treating code as plain text, it understands the grammar of programming languages.

For example, given this Python code:
```python
def greet(name):
    return f"Hello, {name}"
```

Tree-sitter produces a tree like:
```
function_definition
  ├── name: "greet"
  ├── parameters: (name)
  └── body: return statement
```

This is how the app knows exactly where each function starts and ends, what it's called, and what other functions it calls — without manually writing rules for every language.

### 3.5 What is LangChain?

LangChain is a Python library (a collection of pre-built tools) that makes it easier to build applications powered by language models like GPT. It provides:
- **`ChatOpenAI`** — a wrapper to call GPT cleanly
- **`OpenAIEmbeddings`** — to convert text to vectors
- **`PineconeVectorStore`** — to store/search vectors in Pinecone
- **`Document`** — a standard container for a piece of text + metadata
- **Message types** (`SystemMessage`, `HumanMessage`, `AIMessage`) — to manage conversation history

---

## 4. Python Basics Used in This Project

Here is every Python concept used, explained from scratch:

### 4.1 Variables and Constants

```python
TOP_K = 6
GRAPH_DEPTH = 1
```

A variable is a name that holds a value. Writing it in ALL_CAPS is a convention meaning "this is a constant — don't change it." Here `TOP_K = 6` means "retrieve 6 code chunks at a time."

### 4.2 Strings and f-strings

```python
name = "Alice"
greeting = f"Hello, {name}"   # → "Hello, Alice"
```

A string is text wrapped in quotes. An **f-string** (formatted string) starts with `f"..."` and lets you put variable values directly inside `{}` placeholders. The app uses f-strings everywhere to build messages and labels dynamically.

### 4.3 Lists and Dictionaries

```python
# List — ordered collection of items
extensions = [".py", ".js", ".go"]

# Dictionary — key → value pairs (like a lookup table)
languages = {
    ".py":  "python",
    ".js":  "javascript",
    ".go":  "go",
}
languages[".py"]   # → "python"
```

Lists and dictionaries are the most common data structures used throughout this app. The graph itself is a dictionary of dictionaries:

```python
graph = {
    "nodes": { "file.py::MyClass::10": { "name": "MyClass", ... } },
    "edges": { "file.py::MyClass::10": ["file.py::helper::25", ...] }
}
```

### 4.4 Sets

```python
CODE_EXTENSIONS = {".py", ".js", ".ts", ".go"}
```

A set is like a list but each item appears only once, and checking membership (`".py" in CODE_EXTENSIONS`) is extremely fast. It's used here to filter which files to process.

### 4.5 Functions

```python
def validate():
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    ...
```

A `def` block defines a reusable piece of logic you can call by name. Functions can accept inputs (parameters) and return outputs.

### 4.6 Classes and Objects

```python
class Retriever:
    def __init__(self, namespace):
        self.namespaces = [namespace]
        self.stores = {}

    def retrieve(self, query: str) -> list:
        ...
```

A **class** is a blueprint for creating objects. An **object** is a live instance of that blueprint with its own internal state.

- `__init__` is the **constructor** — the code that runs when you create a new object: `r = Retriever("CTI-API")`
- `self` refers to the specific instance of the class — it's how an object refers to its own data
- Methods like `retrieve()` are functions that belong to the class

This app defines three classes: `Symbol` (a parsed code element), `Retriever` (the search engine), and `Responder` (the conversation manager).

### 4.7 Dataclasses

```python
from dataclasses import dataclass, field

@dataclass
class Symbol:
    id: str
    name: str
    kind: str
    file_path: str
    line_start: int
    line_end: int
    source_code: str
    docstring: str = ""
    calls: list = field(default_factory=list)
```

A **dataclass** is a shortcut for creating simple data-holder classes. The `@dataclass` decorator (the `@` line) automatically generates the `__init__` method and other boilerplate. It's the cleanest way to define a structured object that just holds data.

`field(default_factory=list)` means: when no value is provided, create a new empty list (not sharing one list across all instances, which would be a subtle bug).

### 4.8 Decorators (`@`)

```python
@dataclass
class Symbol: ...

@app.route("/api/chat", methods=["POST"])
def chat(): ...
```

A decorator is a function that wraps another function or class to add behaviour. You write it with `@` on the line before. The Flask framework uses `@app.route(...)` to say "when someone visits this URL, call this function."

### 4.9 Imports

```python
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
```

`import` loads external code. Python has a standard library (`os`, `json`, `sys`) and third-party packages installed via `pip`. `from X import Y` imports only a specific item from a module, keeping the namespace clean.

### 4.10 Environment Variables and `.env`

```python
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

Environment variables are values set outside the code — they keep secrets like API keys out of the source code. `os.getenv("KEY")` reads one. The `python-dotenv` library reads a `.env` file and loads its contents as environment variables automatically.

### 4.11 Reading and Writing Files

```python
# Reading
with open("session.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Writing
with open("session.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
```

The `with open(...)` pattern opens a file, runs the indented code, then closes the file automatically — even if an error occurs. `json.load(f)` parses JSON into a Python dictionary. `json.dump(data, f)` converts a dictionary back to JSON text.

### 4.12 List Comprehensions

```python
documents = [_symbol_to_document(s, repo_name) for s in symbols]
repos = [r["repo_name"] for r in data["repos"]]
```

A compact way to build a new list by transforming every item in an existing list. `[expression for item in collection]` is equivalent to a `for` loop that appends to a list, written in one line.

### 4.13 Exception Handling (`try` / `except`)

```python
try:
    fetch_resp = self._index.fetch(ids=neighbor_ids, namespace=ns)
except Exception:
    pass  # graph expansion is best-effort
```

When code might fail (network error, file not found, etc.), you wrap it in `try`. If an error occurs, execution jumps to the `except` block instead of crashing the program. Using `pass` means "silently ignore the error and continue" — appropriate here because graph expansion is a bonus feature, not essential.

### 4.14 Type Hints

```python
def retrieve(self, query: str) -> list:
def build_context(chunks: list, max_chunks: int) -> str:
```

The `:` and `->` annotations describe what types the inputs and output should be. Python doesn't enforce these at runtime, but they make the code much easier to read and allow editors to catch mistakes.

### 4.15 `if __name__ == "__main__"`

```python
if __name__ == "__main__":
    main()
```

This pattern means: "only run `main()` if this file is being run directly (not imported by another file)." It allows the same file to be both a runnable script and an importable module.

### 4.16 BFS — Breadth-First Search with `deque`

```python
from collections import deque

queue = deque([(symbol_id, 0)])
while queue:
    node_id, d = queue.popleft()
    for neighbor in graph["edges"].get(node_id, []):
        queue.append((neighbor, d + 1))
```

A `deque` (double-ended queue) is a list optimised for appending and removing from both ends. BFS is a graph traversal algorithm: starting from a node, visit all its direct neighbours first, then their neighbours, up to a maximum depth. This is how the graph expansion works — starting from a matched code chunk, find what it calls, and what those call, up to `GRAPH_DEPTH` hops.

---

## 5. The Files — One by One

### 5.1 `.env` — Secrets File

```
OPENAI_API_KEY=sk-proj-...
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=repo-chatbot
PINECONE_REGION=us-east-1
```

This file stores secret credentials that must never be shared publicly or committed to version control. It is intentionally listed in `.gitignore`. The `python-dotenv` library reads this file at startup and injects the values as environment variables.

**Why a separate file?** Hard-coding API keys directly in source code is a serious security risk — anyone who reads your code would have access to your (paid) API accounts.

---

### 5.2 `requirements.txt` — Dependency List

```
flask>=3.0.0
openai>=1.0.0
langchain>=0.2.0
langchain-openai>=0.1.0
langchain-pinecone>=0.1.0
langchain-community>=0.2.0
langchain-text-splitters>=0.2.0
pinecone>=3.0.0
tree-sitter>=0.21.0,<0.22.0
tree-sitter-languages>=1.10.0
python-dotenv>=1.0.0
tqdm>=4.0.0
pyyaml>=6.0.0
```

Every line is a Python package that must be installed. The `>=` means "this version or newer." The constraint `<0.22.0` on `tree-sitter` pins it below a version that broke the API. You install everything in one command: `pip install -r requirements.txt`.

**Why a requirements file?** Without it, anyone setting up the project would have to guess which packages are needed and which versions are compatible. This file makes setup reproducible and deterministic.

---

### 5.3 `config.py` — The Settings Dashboard

**Full path:** `repo_chatbot/config.py`

This file centralises every configurable value in the application. Nothing is hardcoded anywhere else — all other files import from here.

```python
import os
import sys
from dotenv import load_dotenv

load_dotenv()  # reads .env into environment variables

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

EMBEDDING_MODEL  = "text-embedding-3-small"
LLM_MODEL        = "gpt-4o-mini"

TOP_K          = 6    # how many chunks to retrieve from vector search
GRAPH_DEPTH    = 1    # how many hops to follow in the graph
MAX_NEIGHBORS  = 3    # max neighbours to follow per node
MAX_CHUNKS     = 12   # max total chunks sent to GPT
```

**The `validate()` function:**

```python
def validate():
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    if missing:
        print("ERROR: ...")
        sys.exit(1)
```

Called at the very start of `ingest.py` and `app.py`. If required keys are missing, the program prints a helpful error message and exits immediately (exit code 1 = error). This prevents confusing failures deep inside the application caused by a missing key.

**Extension sets:**

```python
CODE_EXTENSIONS = {".py", ".js", ".mjs", ".ts", ".go", ".java"}
TEXT_EXTENSIONS = {".json", ".yml", ".yaml", ".md", ".html"}
ALL_EXTENSIONS  = CODE_EXTENSIONS | TEXT_EXTENSIONS
```

The `|` operator on sets produces their union (all items from both). Code extensions get tree-sitter parsing; text extensions get simpler text splitting.

**Why separate code from text?** Code has structure (functions, classes, call relationships) that tree-sitter can extract. Text files like JSON or Markdown are just content — they need simple splitting, not AST parsing.

**`IGNORED_DIRS`:**

```python
IGNORED_DIRS = {
    "node_modules", ".git", "__pycache__",
    ".venv", "venv", "dist", "build",
    ".next", "vendor",
}
```

These directories contain generated or dependency code, not the developer's actual logic. Indexing them would add enormous amounts of irrelevant content and waste API credits.

---

### 5.4 `parser.py` — The Code Reader

**Full path:** `repo_chatbot/parser.py`

This is the most complex file. It is responsible for two things:
1. Reading every file in a repository and extracting meaningful code units (the `Symbol` objects)
2. Building the relationship graph from those symbols

#### The `Symbol` Dataclass

```python
@dataclass
class Symbol:
    id: str           # unique ID: "path/file.py::ClassName::42"
    name: str         # "ClassName" or "my_function"
    kind: str         # "function", "class", "method", or "module"
    file_path: str    # relative path within the repo
    line_start: int
    line_end: int
    source_code: str  # the actual code text
    docstring: str    # the documentation comment, if any
    signature: str    # the first line (e.g., "def greet(name):")
    calls: list       # names of functions this symbol calls
    parent_class: str # if this is a method, the class it belongs to
```

Every function, class, and method in the codebase becomes one `Symbol` object. This is the fundamental data unit of the entire application.

**Why extract symbols instead of just splitting files into paragraphs?**

A function is a natural, meaningful unit of code. If you split a file every 500 characters, you might cut a function in half, losing the context that ties lines together. Extracting whole functions means each chunk is semantically complete and self-explanatory.

#### Tree-sitter Parsing

```python
_FUNCTION_TYPES = {
    "python":     {"function_definition"},
    "javascript": {"function_declaration", "arrow_function", ...},
    "go":         {"function_declaration"},
    ...
}
```

Different languages use different syntax for functions. Tree-sitter's output uses language-specific node type names. These dictionaries map each language to the node types that represent functions, classes, methods, and function calls.

**The `_walk` function** recursively traverses the tree-sitter parse tree:
- When it finds a class node → create a `Symbol(kind="class")`
- When it finds a function inside a class → create a `Symbol(kind="method")`
- When it finds a top-level function → create a `Symbol(kind="function")`
- It records `calls` by scanning within each function for call expressions

#### Handling Different File Types

| File type | Strategy |
|-----------|----------|
| `.py`, `.js`, `.ts`, `.go`, `.java` | Parse with tree-sitter; extract functions/classes |
| `.json` (small) | Treat as one module |
| `.json` (large) | Split by top-level keys; each key becomes a chunk |
| `.md` | Split by markdown headings using `MarkdownTextSplitter` |
| `.yml`/`.yaml` | Split by top-level keys |
| `.html` | Split into 1000-character chunks with 100-character overlap |

**Why different strategies?** Treating all files identically would either create chunks too small to be useful or chunks too large for the embedding model (which has a character limit). Each format has a natural "unit" — JSON keys, Markdown sections, code functions — and the parser uses those natural divisions.

#### The Graph: `build_graph()`

```python
def build_graph(symbols: list) -> dict:
    graph = {"nodes": {}, "edges": {}}
    name_index = {}   # function name → [list of IDs with that name]

    # Pass 1: register all nodes
    for sym in symbols:
        graph["nodes"][sym.id] = { "name": sym.name, "kind": sym.kind, ... }
        graph["edges"][sym.id] = []
        name_index.setdefault(sym.name, []).append(sym.id)

    # Pass 2: build edges from call relationships
    for sym in symbols:
        for called_name in sym.calls:
            for target_id in name_index.get(called_name, []):
                if target_id != sym.id:
                    graph["edges"][sym.id].append(target_id)

    return graph
```

**Two-pass approach:** The first pass registers all nodes so every symbol has an entry. The second pass builds edges — for each function's list of names it calls, look up which node IDs have that name and draw an edge.

**Why `name_index`?** When parsing `function_A` that calls `"validate"`, we don't know the full ID of `validate` yet. The `name_index` dictionary maps plain names to their full IDs, making the lookup in pass 2 trivial.

#### Graph Traversal: `get_neighbors()`

```python
def get_neighbors(graph, symbol_id, depth, max_per_node):
    visited = {symbol_id}
    result  = []
    queue   = deque([(symbol_id, 0)])

    while queue:
        node_id, d = queue.popleft()
        if d >= depth:
            continue
        for neighbor in graph["edges"].get(node_id, [])[:max_per_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                result.append(neighbor)
                queue.append((neighbor, d + 1))
    return result
```

Standard BFS (Breadth-First Search):
- `visited` prevents visiting the same node twice (avoids infinite loops in circular call chains)
- `d` tracks depth; stops at `GRAPH_DEPTH = 1` (only immediate neighbours by default)
- `max_per_node` caps how many neighbours are followed per node (avoids exploding fan-out)

**Why depth 1 and max 3?** Empirically, going deeper pulls in too many loosely related functions, which dilutes the context quality. One hop usually captures the most relevant related code.

---

### 5.5 `ingest.py` — The Uploader

**Full path:** `repo_chatbot/ingest.py`

This is the setup script. You run it once (or whenever the repositories change) on the command line:

```bash
python ingest.py ./CTI-API ./Service-Portal-API
```

#### Flow:

1. **Parse arguments** — `argparse` reads the two folder paths from the command line
2. **Create embeddings client** — connects to OpenAI's embedding model
3. **Create Pinecone client** — connects to the vector database
4. **For each repository:**
   - Call `parse_repo()` to extract all `Symbol` objects
   - Call `build_graph()` to build the relationship map
   - Save the graph to `data/{repo_name}_graph.json`
   - Convert each `Symbol` to a LangChain `Document`
   - Upload in batches of 100 to Pinecone
5. **Save session.json** — records what repos are loaded (used by the web app)

#### Converting a Symbol to a Document

```python
def _symbol_to_document(sym, repo_name):
    page_content = (
        f"File: {sym.file_path}\n"
        f"Type: {sym.kind} named {sym.name}\n"
        f"Signature: {sym.signature}\n"
        f"Description: {sym.docstring}\n"
        f"Calls: {', '.join(sym.calls)}\n"
        f"\n{sym.source_code}"
    )
    metadata = { "chunk_id": sym.id, "file_path": sym.file_path, ... }
    return Document(page_content=page_content, metadata=metadata)
```

The `page_content` is the text that gets converted into a vector. It is deliberately structured to include the file path, type, signature, docstring, and source code — giving the embedding model maximum context about what this chunk is.

The `metadata` is stored alongside the vector in Pinecone and returned with search results, allowing the app to display file paths and line numbers.

**Why not just embed the raw source code?** Including the file path, function name, and docstring in the embedded text dramatically improves search quality. When you search for "how does authentication work," the embedding of "File: auth/middleware.go, Function: authenticate, Description: validates JWT tokens..." will score much higher than just the raw Go code, because the description is in natural language.

#### Batching Uploads

```python
batch_size = 100
for i in range(0, len(documents), batch_size):
    vector_store.add_documents(documents[i:i + batch_size])
```

APIs have rate limits (maximum requests per minute). Uploading 1,570 documents individually would quickly hit those limits. Batching 100 at a time groups them into fewer API calls. The `range(0, len(documents), batch_size)` generates indices 0, 100, 200, 300... and `documents[i:i+100]` slices the next batch.

---

### 5.6 `retriever.py` — The Smart Searcher

**Full path:** `repo_chatbot/retriever.py`

This file is used during every chat message. Its job is to find the most relevant code chunks for a given question.

#### The `Retriever` Class

```python
class Retriever:
    def __init__(self, namespace):
        self.namespaces = [namespace] if isinstance(namespace, str) else namespace
        # Connect to Pinecone (one store per repo namespace)
        self.stores = { ns: PineconeVectorStore(..., namespace=ns) for ns in self.namespaces }
        # Load the relationship graphs from disk
        self.graphs = { ns: load_graph(f"data/{ns}_graph.json") for ns in self.namespaces }
```

**Namespaces:** Pinecone organises vectors in namespaces — essentially labelled buckets. Each repository gets its own namespace (`"CTI-API"`, `"Service-Portal-API"`). This means you can search one or both simultaneously without them interfering.

**Why load graphs from disk?** The graph files (JSON) were generated during ingestion and saved locally. Loading them is fast (milliseconds). There's no need to store the graph in Pinecone — it lives alongside the application.

#### The `retrieve()` Method — Two-Stage Search

**Stage 1: Semantic (vector) search**

```python
raw_results = []
for ns in self.namespaces:
    pairs = self.stores[ns].similarity_search_with_score(query, k=config.TOP_K)
    for doc, score in pairs:
        raw_results.append((score, doc, ns))

raw_results.sort(key=lambda x: x[0], reverse=True)
seeds = raw_results[:config.TOP_K]
```

For each namespace, ask Pinecone "what are the 6 most similar vectors to this query?" Then merge all results, sort by score (highest first), and keep only the top 6. These are called "seeds" — the starting points for graph expansion.

`similarity_search_with_score` returns a list of `(Document, score)` pairs. The score is a similarity number (higher = more relevant).

`sort(key=lambda x: x[0], reverse=True)` sorts the list by the first element of each tuple (the score), in descending order (highest first). The `lambda` is an anonymous function used inline.

**Stage 2: Graph expansion**

```python
for score, doc, ns in seeds:
    chunk_id = doc.metadata.get("chunk_id", "")
    # Store the seed chunk
    chunks_by_id[chunk_id] = { ..., "retrieval_type": "direct" }

    # Find neighbours in the graph
    graph = self.graphs.get(ns, {})
    neighbor_ids = get_neighbors(graph, chunk_id, depth=1, max_per_node=3)

    # Fetch neighbour vectors from Pinecone by ID
    fetch_resp = self._index.fetch(ids=neighbor_ids, namespace=ns)
    for nid, vec in fetched.items():
        if nid not in chunks_by_id:
            chunks_by_id[nid] = { ..., "retrieval_type": "graph_neighbor" }
```

For each seed chunk: look up its neighbours in the local graph, then fetch those chunks directly from Pinecone by their IDs (not by similarity — by exact ID lookup). This is much more targeted than running another similarity search.

`chunks_by_id` is a dictionary keyed by chunk ID. Using a dictionary automatically deduplicates — if the same function is a neighbour of two different seeds, it only appears once.

**Why graph expansion on top of vector search?**

Vector search is great at finding the function that directly answers your question. But code is written in layers — the function you find often delegates to helper functions, which are also essential context. Without graph expansion, GPT sees only the surface; with it, GPT sees the full call chain.

#### `build_context()` — Formatting for GPT

```python
def build_context(chunks, max_chunks):
    direct   = [c for c in chunks if c["retrieval_type"] == "direct"]
    neighbor = [c for c in chunks if c["retrieval_type"] == "graph_neighbor"]

    selected = direct[:max_chunks]
    remaining = max_chunks - len(selected)
    selected += neighbor[:remaining]

    selected.sort(key=lambda c: (c["repo_name"], c["file_path"], c["line_start"]))

    parts = ["=== CODEBASE CONTEXT ===\n"]
    for i, c in enumerate(selected, 1):
        header = f"[{i}] [{c['repo_name']}] {c['file_path']} | {c['kind']}: {c['name']} ..."
        parts.append(header)
        parts.append("---")
        parts.append(c["source_code"])
    return "\n".join(parts)
```

Direct results (from vector search) are prioritised over graph neighbours. The combined list is sorted by `(repo_name, file_path, line_start)` — a tuple sort — so chunks from the same file appear together in reading order. This makes the context block easier for GPT to follow.

**Why sort by file and line?** If GPT receives chunks in random order it may get confused about which code follows which. Sorting by file and line produces a context that reads like a code file, helping GPT reason about flow.

---

### 5.7 `responder.py` — The Conversationalist

**Full path:** `repo_chatbot/responder.py`

This file manages the conversation — tracking history, rewriting ambiguous questions, and calling GPT.

#### System Prompts

```python
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
```

A **system prompt** is a special instruction given to GPT at the start of every conversation. It sets the persona and rules. Key constraints here:

- *"using only the source code provided"* — prevents GPT from hallucinating facts about code it hasn't seen
- *"If the context does not contain the answer, say so"* — prevents confident wrong answers
- *"Always mention the file path and function name"* — makes answers actionable and verifiable

#### The `Responder` Class

```python
class Responder:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, ...)
        self.rewriter = ChatOpenAI(model="gpt-4o-mini", temperature=0, ...)
        self.history: list = []
        self.last_sources: list = []
```

Two separate LLM instances are created:
- `self.llm` — answers questions, with `temperature=0.2` (slightly creative, good for explanations)
- `self.rewriter` — rewrites questions, with `temperature=0` (deterministic, exact, no creativity wanted)

**What is temperature?** Temperature controls randomness. At `0`, the model always picks the most likely next word — fully deterministic and precise. At `1`, it picks more surprisingly. For rewriting a question you want zero ambiguity; for explaining code you want a natural, readable tone.

#### The Rewriter — Why It Exists

```python
def _rewrite_question(self, question: str) -> str:
    if not self.history:
        return question   # first question: no history, no need to rewrite

    recent = self.history[-8:]   # last 4 turns (each turn = 2 messages)
    messages = [SystemMessage(content=_REWRITE_SYSTEM)]
    messages += recent
    messages.append(HumanMessage(content=question))
    response = self.rewriter.invoke(messages)
    return response.content.strip()
```

**The problem it solves:** Conversations use pronouns and references.

- Turn 1: "How does the login function work?"
- Turn 2: "What does **it** return?"

The word "it" is meaningless without context. If you send "What does it return?" directly to the vector search, you'll get random results. The rewriter transforms it to: *"What does the login function return?"* — which then finds the right code.

**Why a separate LLM call for rewriting?** Using the same GPT call for both rewriting and answering would conflate two different tasks. The rewriter is a precision instrument (temperature=0) that does one mechanical job. Separating them also means if rewriting fails, the original question is used as a fallback.

**Only last 8 messages (4 turns):** Full history would make the rewriter prompt very long and costly. The last 4 turns almost always contain all the necessary context for resolving pronouns.

#### The `chat()` Method

```python
def chat(self, question: str, namespace) -> dict:
    rewritten = self._rewrite_question(question)  # Step 1: clarify

    retriever = Retriever(namespace)
    chunks    = retriever.retrieve(rewritten)      # Step 2: find relevant code
    context   = build_context(chunks, MAX_CHUNKS)  # Step 3: format it

    messages = [SystemMessage(content=_SYSTEM_PROMPT)]
    messages += self.history           # full conversation history
    messages.append(HumanMessage(
        content=f"Context:\n{context}\n\nQuestion: {question}"
    ))

    response = self.llm.invoke(messages)  # Step 4: ask GPT
    answer   = response.content

    self.history.append(HumanMessage(content=question))  # Step 5: save to history
    self.history.append(AIMessage(content=answer))

    return { "answer": answer, "sources": chunks, "rewritten_query": rewritten }
```

Notice: the **rewritten** question is used for retrieval, but the **original** question is sent to GPT and stored in history. This is intentional — GPT sees what the user actually typed, which sounds more natural, while the search uses the clarified version which finds better results.

The context is injected directly into the user message, not the system prompt, so it is fresh every turn (different questions need different context).

---

### 5.8 `app.py` — The Web Server

**Full path:** `repo_chatbot/app.py`

Flask is a lightweight Python web framework. This file is the HTTP layer — it exposes three URL endpoints that the browser calls.

```python
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static")
responder = Responder()   # one shared responder instance for the whole server
```

**One shared `Responder`:** Because `Responder` stores conversation history (`self.history`), there must be only one instance for the server — otherwise each request would get a fresh, memoryless instance and the conversation would break.

#### The Three Routes

**Route 1: Serve the UI**
```python
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")
```
When someone opens `http://localhost:5000`, Flask sends back the `index.html` file from the `static/` folder.

**Route 2: Check status**
```python
@app.route("/api/status")
def status():
    with open(config.SESSION_FILE, "r") as f:
        data = json.load(f)
    repos = [{"repo_name": r["repo_name"], "chunk_count": r["chunk_count"]} for r in data["repos"]]
    return jsonify({"loaded": True, "repos": repos})
```
Returns a JSON object describing which repositories are loaded. The browser calls this on startup to populate the header pills and repo selector buttons.

**Route 3: Handle a chat message**
```python
@app.route("/api/chat", methods=["POST"])
def chat():
    body    = request.get_json(force=True)
    message = body.get("message", "").strip()
    namespace = body.get("namespace", "both")

    if namespace == "both":
        ns = [r["repo_name"] for r in responder.repos]
    else:
        ns = namespace

    result = responder.chat(message, ns)
    return jsonify({ "answer": result["answer"], "sources": result["sources"], ... })
```
`methods=["POST"]` means this route only accepts POST requests (the browser sends the message in the request body, not the URL). `request.get_json()` parses the JSON body into a Python dictionary. `jsonify()` converts a Python dictionary into a JSON HTTP response.

**Route 4: Clear conversation**
```python
@app.route("/api/clear", methods=["POST"])
def clear():
    responder.clear()
    return jsonify({"status": "ok"})
```
Resets `self.history = []` in the Responder. The browser clears the chat bubbles on its side too.

```python
if __name__ == "__main__":
    app.run(debug=True, port=5000)
```
Starts the development server on port 5000. `debug=True` enables auto-reload when files change and shows detailed error pages — only appropriate for development, not a public deployment.

---

### 5.9 `static/index.html` — The User Interface

**Full path:** `repo_chatbot/static/index.html`

This single file is the entire browser-side application. It contains HTML structure, CSS styling, and JavaScript logic in one place.

#### The Dark Theme (CSS Variables)

```css
:root {
  --bg:      #0f1117;
  --surface: #1a1d27;
  --accent:  #4f8ef7;
  --text:    #e2e8f0;
}
```

CSS custom properties (variables) defined on `:root` are available everywhere. Changing `--accent` in one place updates all buttons and focused inputs — a clean, maintainable approach to theming.

#### The Layout

The page is a vertical flex column: header → repo selector bar → chat area (scrollable) → input bar. The chat area has `flex: 1` meaning it takes all remaining space. This creates the "sticky header and footer, scrollable middle" pattern typical of chat apps.

#### How a Message is Sent (JavaScript)

```javascript
async function sendMessage() {
    const text = msgEl.value.trim();
    if (!text) return;

    appendBubble("user", text);           // show user message immediately
    const thinkEl = appendThinking();     // show "Thinking..." animation

    const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, namespace: selectedNs }),
    });
    const data = await resp.json();

    thinkEl.remove();                     // remove animation
    appendBubble("bot", data.answer, true); // show GPT's answer
    appendSources(wrap, data.sources, ...); // show source references
}
```

`async`/`await` is JavaScript's way of handling operations that take time (like network requests) without freezing the page. `fetch()` makes an HTTP request to the Flask server. The user sees their message and a "Thinking..." animation immediately, then the answer appears when it arrives.

#### Markdown Rendering

```javascript
import { marked } from "marked";  // loaded from CDN

bubble.innerHTML = marked.parse(data.answer);
```

GPT's answers often contain markdown — `**bold**`, ` ```code blocks``` `, bullet lists. The `marked` library (loaded from a CDN, no installation needed) converts markdown to HTML so it renders properly instead of showing raw asterisks.

#### Sources Panel

Each bot answer has a collapsible "📎 N sources" link. Clicking it reveals the list of code chunks that were used to generate the answer, differentiated by retrieval type:
- Regular style: direct vector search result
- Dimmed "↗" prefix: graph expansion neighbour

This transparency helps users verify the answer came from the right code.

#### The Rewrite Note

```javascript
if (rewritten && rewritten !== original) {
    note.textContent = `🔍 Searched as: ${rewritten}`;
}
```

When the rewriter changed the question, a small italic note shows what the actual search query was. This demystifies the AI — the user can see "it turned 'what does it return' into 'what does the login function return'."

---

### 5.10 `data/` — Saved State

The `data/` directory holds three types of files:

| File | Created by | Used by |
|------|-----------|---------|
| `session.json` | `ingest.py` | `app.py`, `responder.py` |
| `CTI-API_graph.json` | `ingest.py` via `parser.py` | `retriever.py` |
| `Service-Portal-API_graph.json` | `ingest.py` via `parser.py` | `retriever.py` |

**`session.json`:**
```json
{
  "repos": [
    { "repo_name": "CTI-API", "repo_root": "/path/to/CTI-API", "chunk_count": 1570 },
    { "repo_name": "Service-Portal-API", "repo_root": "...", "chunk_count": 40 }
  ]
}
```
A simple record of what was ingested. The web app reads this to know what repos exist and how many chunks each has.

**`*_graph.json`:**
```json
{
  "nodes": {
    "auth/middleware.go::authenticate::12": {
      "name": "authenticate",
      "kind": "function",
      "file_path": "auth/middleware.go",
      "line_start": 12,
      "line_end": 45
    }
  },
  "edges": {
    "auth/middleware.go::authenticate::12": [
      "services/token.go::validateToken::8",
      "db/users.go::lookupUser::22"
    ]
  }
}
```

Nodes are all identified functions/classes. Edges map each function to the IDs of functions it calls.

---

## 6. Key Design Decisions Explained

### Why Graph + Vector Search together?

Vector search alone has a critical blind spot: **it finds the function you asked about, but not the functions it depends on**. If you ask "how does authentication work?" and the entry point is `authenticate()`, you also need `validateToken()` and `lookupUser()` — the functions it calls — to give GPT enough context for a complete answer.

The graph provides exactly this: a pre-computed map of call relationships that can be traversed in milliseconds. Combining both approaches gives the best of both worlds — semantic relevance from vector search, structural completeness from the graph.

### Why tree-sitter over simple regex or text splitting?

Regular expressions can find `def function_name(` but fail on:
- Docstrings that span multiple lines
- Nested functions
- Language-specific syntax differences
- Extracting what a function actually *calls*

Tree-sitter builds a real syntactic tree, making all of this trivially accurate. The trade-off is complexity (more code, more dependencies), but the accuracy gain is essential for a tool that claims to understand code structure.

### Why the question rewriter?

In a multi-turn conversation, pronouns and references like "it", "that function", "the one you mentioned" are extremely common. Sending such queries to a vector database produces garbage results because those words have no specific meaning in isolation. The rewriter transforms ambiguous follow-up questions into standalone queries — a technique called **query rewriting** — and is a standard component of production RAG (Retrieval-Augmented Generation) systems.

A key subtlety: the rewriter uses `temperature=0` (completely deterministic) while the answerer uses `temperature=0.2` (slightly flexible). Rewriting is a mechanical transformation that needs no creativity. Answering benefits from natural language variety.

### Why `gpt-4o-mini` instead of `gpt-4o`?

`gpt-4o-mini` is ~10x cheaper and 3x faster than `gpt-4o` while being sufficient for code explanation tasks. Since every question involves multiple API calls (rewriter + answerer, plus the embedding), cost compounds quickly. For a tool used frequently across a large codebase, cost matters.

### Why Pinecone instead of a local vector database?

Options like FAISS or ChromaDB run locally with no external service. However, Pinecone's free tier is generous, and it provides:
- **Namespaces** — clean separation of repos without running two databases
- **Fetch by ID** — essential for graph expansion (fetching neighbours by exact ID, not similarity)
- **Managed infrastructure** — no memory management or index persistence to worry about

The `fetch by ID` feature is particularly critical: after vector search finds the seeds, the graph expansion needs to retrieve specific chunks by their exact IDs — Pinecone's `.fetch()` method does this efficiently.

### Why inject context into the `HumanMessage` rather than the `SystemMessage`?

The system prompt sets permanent rules (persona, constraints). Context is per-question — it changes with every turn. Injecting it into the human message keeps the system prompt clean and ensures each question gets fresh, relevant context rather than accumulating stale context from previous questions.

### Why `MAX_CHARS = 3000` per chunk?

OpenAI's embedding model (`text-embedding-3-small`) accepts up to ~8,000 tokens. However, embedding very long texts dilutes the semantic signal — the vector ends up representing a mix of many concepts rather than one focused idea. 3,000 characters (roughly 750 words) keeps each chunk focused enough for precise similarity matching.

### Why batch size of 100 for Pinecone uploads?

Pinecone's API has a maximum batch size of 100 vectors per upsert request. Setting exactly 100 maximises throughput while respecting the API limit.

---

## 7. The Full Journey of One Question

Let's trace exactly what happens when you type: **"What does the login function return?"**

1. **Browser** — `sendMessage()` captures the text, shows it as a user bubble, shows "Thinking…"

2. **Browser** — `fetch("/api/chat", { method: "POST", body: JSON.stringify({message: "...", namespace: "both"}) })` sends an HTTP POST to Flask

3. **Flask `app.py`** — `chat()` function is invoked. `namespace = "both"` → `ns = ["CTI-API", "Service-Portal-API"]`

4. **`responder.py` `_rewrite_question()`** — If this is a follow-up question, GPT rewrites "What does it return?" → "What does the login function return?" (if "it" referred to login from before). Otherwise the question passes through unchanged.

5. **`retriever.py` `retrieve()`:**
   - Pinecone converts "What does the login function return?" into a 1536-dimension vector
   - Pinecone searches both namespaces, returns top 6 matching chunks (e.g., `auth/login.py::login::45`)
   - Graph expansion: look up `auth/login.py::login::45` in the graph → neighbours: `[token.py::create_jwt::12, db.py::save_session::88]`
   - Fetch those two neighbours from Pinecone by ID
   - Total: ~8 chunks, deduped, tagged as "direct" or "graph_neighbor"

6. **`retriever.py` `build_context()`** — Prioritises direct results, fills remaining slots with neighbours, sorts by file/line, formats into a text block like:

   ```
   === CODEBASE CONTEXT ===
   [1] [CTI-API] auth/login.py | function: login (lines 45-78)
   ---
   def login(username, password):
       ...
       return { "token": create_jwt(user_id), "expires": ... }
   ```

7. **`responder.py` `chat()`** — Sends to GPT:
   - SystemMessage: (persona + rules)
   - Previous history messages
   - HumanMessage: `"Context:\n[...code...]\n\nQuestion: What does the login function return?"`

8. **GPT-4o-mini** responds:
   > "The `login` function in `auth/login.py` (lines 45–78) returns a dictionary containing two keys: `token` (a JWT string created by `create_jwt`) and `expires` (a timestamp)."

9. **`responder.py`** — saves question and answer to `self.history`, returns `{answer, sources, rewritten_query}`

10. **Flask** — `jsonify(result)` sends the JSON back to the browser

11. **Browser** — removes "Thinking…", renders the answer as markdown, adds collapsible sources panel showing the 8 chunks used

Total time: approximately 3–6 seconds (dominated by the two GPT API calls).

---

## 8. How to Build This Yourself — Step by Step

### Prerequisites

- Python 3.10 or newer installed
- An OpenAI account with an API key (paid, or free trial)
- A Pinecone account (free tier is sufficient)
- Two code repositories you want to chat with

### Step 1: Set up Pinecone

1. Sign up at [pinecone.io](https://pinecone.io)
2. Create a new index:
   - Name: `repo-chatbot`
   - Dimensions: `1536`
   - Metric: `cosine`
3. Copy your API key from the Pinecone dashboard

### Step 2: Create the project folder

```bash
mkdir my-chatbot
cd my-chatbot
```

### Step 3: Create a virtual environment

A virtual environment isolates this project's packages from other Python projects on your machine.

```bash
python -m venv .venv
source .venv/bin/activate      # on Mac/Linux
# .venv\Scripts\activate       # on Windows
```

### Step 4: Create `requirements.txt`

Copy the requirements listed in section 5.2 into a file called `requirements.txt`, then:

```bash
pip install -r requirements.txt
```

### Step 5: Create `.env`

```
OPENAI_API_KEY=your-openai-key-here
PINECONE_API_KEY=your-pinecone-key-here
PINECONE_INDEX_NAME=repo-chatbot
PINECONE_REGION=us-east-1
```

### Step 6: Create all Python files

Create these files in order (each one depends on the previous):
1. `config.py` — settings and validation
2. `parser.py` — code extraction and graph building
3. `ingest.py` — upload to Pinecone
4. `retriever.py` — search logic
5. `responder.py` — conversation management
6. `app.py` — web server

Create a `static/` folder and put `index.html` inside it.

### Step 7: Run ingestion

```bash
python ingest.py /path/to/your/repo1 /path/to/your/repo2
```

This takes 1–10 minutes depending on repository size. You'll see progress printed to the terminal:
```
Processing CTI-API...
  Found 1570 symbols
  Graph saved: 1570 nodes, 3241 edges
  Uploaded 1570 chunks to Pinecone
```

### Step 8: Start the web server

```bash
python app.py
```

Open `http://localhost:5000` in your browser. You should see the chat interface with your two repository names in the header.

### Step 9: Ask questions

Click "Both" or select a specific repository, then type any question about the codebase. The first response may be slightly slow (cold start); subsequent ones are faster.

---

### Common Issues and Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| `ERROR: OPENAI_API_KEY not set` | `.env` file missing or misnamed | Check the file is named exactly `.env` (not `env.txt`) |
| `No repos loaded — run ingest.py first` | `data/session.json` doesn't exist | Run `python ingest.py ...` first |
| Ingestion is slow | Large repositories | Normal; each symbol requires an API call. Batching helps. |
| Answers reference wrong code | Namespace mismatch | Ensure namespace in chat matches the repo name used during ingestion |
| Graph expansion returns nothing | Graph file missing | Re-run `ingest.py`; check `data/` folder contains `*_graph.json` files |

---

*This document was written to be fully self-contained. If you have read it from the beginning, you now have a complete mental model of the application — from the raw Python language features used to the architectural decisions made and why.*
