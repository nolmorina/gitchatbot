import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Required — loaded from .env
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")

# Optional — loaded from .env with defaults
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "repo-chatbot")
PINECONE_REGION     = os.getenv("PINECONE_REGION", "us-east-1")

# Model settings
EMBEDDING_MODEL     = "text-embedding-3-small"
LLM_MODEL           = "gpt-5-mini"

# Retrieval settings
TOP_K               = 6
GRAPH_DEPTH         = 1
MAX_NEIGHBORS       = 3
MAX_CHUNKS          = 12

# File handling
DATA_DIR            = "./data"
SESSION_FILE        = "./data/session.json"

# Which file types to process
CODE_EXTENSIONS     = {".py", ".js", ".mjs", ".ts", ".go", ".java"}
TEXT_EXTENSIONS     = {".json", ".yml", ".yaml", ".md", ".html"}
ALL_EXTENSIONS      = CODE_EXTENSIONS | TEXT_EXTENSIONS

# Folders to always skip
IGNORED_DIRS        = {
    "node_modules", ".git", "__pycache__",
    ".venv", "venv", "dist", "build",
    ".next", "vendor",
}

# tree-sitter language name for each code extension
LANGUAGES           = {
    ".py":   "python",
    ".js":   "javascript",
    ".mjs":  "javascript",
    ".ts":   "typescript",
    ".go":   "go",
    ".java": "java",
}

# Large JSON/YAML files: split after this many lines
BIG_FILE_LINE_LIMIT = 200

# Hard cap: no single chunk sent to embedding can exceed this
MAX_CHARS           = 3000


def validate():
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    if missing:
        print("ERROR: The following required environment variables are not set:")
        for key in missing:
            print(f"  - {key}")
        print("Please set them in your .env file before running.")
        sys.exit(1)
