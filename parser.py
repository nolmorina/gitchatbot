import os
import json
import yaml
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter

import config

# ── Parser cache — one parser per language ───────────────────────────────────
_parser_cache: dict = {}


def _get_ts_parser(language: str):
    if language not in _parser_cache:
        from tree_sitter_languages import get_parser
        _parser_cache[language] = get_parser(language)
    return _parser_cache[language]


# ── Per-language tree-sitter node type sets ───────────────────────────────────
_FUNCTION_TYPES = {
    "python":     {"function_definition"},
    "javascript": {"function_declaration", "function_expression", "arrow_function",
                   "generator_function_declaration", "generator_function"},
    "typescript": {"function_declaration", "function_expression", "arrow_function",
                   "generator_function_declaration", "generator_function"},
    "go":         {"function_declaration"},
    "java":       set(),   # java uses method_declaration inside class_body
}
_CLASS_TYPES = {
    "python":     {"class_definition"},
    "javascript": {"class_declaration", "class_expression"},
    "typescript": {"class_declaration", "class_expression"},
    "go":         set(),
    "java":       {"class_declaration", "interface_declaration", "enum_declaration"},
}
_METHOD_TYPES = {
    "python":     {"function_definition"},
    "javascript": {"method_definition"},
    "typescript": {"method_definition"},
    "go":         {"method_declaration"},
    "java":       {"method_declaration", "constructor_declaration"},
}
_CALL_TYPES = {
    "python":     {"call"},
    "javascript": {"call_expression"},
    "typescript": {"call_expression"},
    "go":         {"call_expression"},
    "java":       {"method_invocation"},
}


# ── Symbol dataclass ──────────────────────────────────────────────────────────
@dataclass
class Symbol:
    id: str
    name: str
    kind: str           # "function" | "class" | "method" | "module"
    file_path: str
    line_start: int
    line_end: int
    source_code: str
    docstring: str = ""
    signature: str = ""
    calls: list = field(default_factory=list)
    parent_class: str = ""


# ── Tree-sitter helpers ───────────────────────────────────────────────────────
def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _name_of(node, src: bytes) -> str:
    """First identifier child = the declared name."""
    for child in node.children:
        if child.type in ("identifier", "property_identifier", "field_identifier",
                          "name", "type_identifier"):
            return _text(child, src)
    return "<anonymous>"


def _signature_of(node, src: bytes) -> str:
    return _text(node, src).split("\n")[0].strip()


def _docstring_of(node, src: bytes) -> str:
    """First string literal in the body block."""
    body = None
    for child in node.children:
        if child.type in ("block", "suite", "statement_block",
                          "declaration_list", "class_body"):
            body = child
            break
    target = body if body is not None else node
    for child in target.children:
        if child.type == "expression_statement":
            for gc in child.children:
                if gc.type in ("string", "string_literal"):
                    return _text(gc, src).strip("\"'` \n")
        if child.type in ("string", "string_literal"):
            return _text(child, src).strip("\"'` \n")
    return ""


def _calls_in(node, src: bytes, call_types: set) -> list:
    """All function names called within this node's subtree."""
    names = set()
    queue = deque([node])
    while queue:
        n = queue.popleft()
        if n.type in call_types and n.children:
            callee = n.children[0]
            if callee.type in ("attribute", "member_expression", "field_access",
                               "selector_expression"):
                for c in callee.children:
                    if c.type in ("identifier", "property_identifier"):
                        names.add(_text(c, src))
            elif callee.type == "identifier":
                names.add(_text(callee, src))
        queue.extend(n.children)
    return list(names)


def _cap(text: str) -> str:
    if len(text) > config.MAX_CHARS:
        return text[:config.MAX_CHARS] + "\n... [truncated]"
    return text


def _make_module(rel_path: str, raw: str) -> "Symbol":
    lines = raw.splitlines()
    return Symbol(
        id=f"{rel_path}::module::1",
        name=os.path.basename(rel_path),
        kind="module",
        file_path=rel_path,
        line_start=1,
        line_end=max(len(lines), 1),
        source_code=_cap(raw),
    )


# ── Tree walker ───────────────────────────────────────────────────────────────
def _walk(tree, src: bytes, rel_path: str, language: str) -> list:
    fn_types   = _FUNCTION_TYPES.get(language, set())
    cls_types  = _CLASS_TYPES.get(language, set())
    mth_types  = _METHOD_TYPES.get(language, set())
    call_types = _CALL_TYPES.get(language, set())
    results = []

    def visit(node, current_class: Optional[str] = None):
        if node.type in cls_types:
            name  = _name_of(node, src)
            start = node.start_point[0] + 1
            end   = node.end_point[0] + 1
            results.append(Symbol(
                id=f"{rel_path}::{name}::{start}",
                name=name, kind="class",
                file_path=rel_path, line_start=start, line_end=end,
                source_code=_cap(_text(node, src)),
                docstring=_docstring_of(node, src),
                signature=_signature_of(node, src),
                calls=[], parent_class="",
            ))
            for child in node.children:
                visit(child, current_class=name)

        elif (current_class and node.type in mth_types) or \
             (not current_class and node.type in fn_types):
            name  = _name_of(node, src)
            start = node.start_point[0] + 1
            end   = node.end_point[0] + 1
            kind  = "method" if current_class else "function"
            results.append(Symbol(
                id=f"{rel_path}::{name}::{start}",
                name=name, kind=kind,
                file_path=rel_path, line_start=start, line_end=end,
                source_code=_cap(_text(node, src)),
                docstring=_docstring_of(node, src),
                signature=_signature_of(node, src),
                calls=_calls_in(node, src, call_types),
                parent_class=current_class or "",
            ))
            # do not recurse into nested functions

        else:
            for child in node.children:
                visit(child, current_class=current_class)

    visit(tree.root_node)
    return results


# ── Public: extract_symbols ───────────────────────────────────────────────────
def extract_symbols(abs_path: str, rel_path: str, extension: str) -> list:
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except Exception:
        return []

    # ── Code files ──────────────────────────────────────────────────────────
    if extension in config.CODE_EXTENSIONS:
        language = config.LANGUAGES.get(extension)
        if not language:
            return [_make_module(rel_path, raw)]
        try:
            parser = _get_ts_parser(language)
            tree   = parser.parse(raw.encode("utf-8"))
            syms   = _walk(tree, raw.encode("utf-8"), rel_path, language)
            return syms if syms else [_make_module(rel_path, raw)]
        except Exception:
            return [_make_module(rel_path, raw)]

    # ── Text files ───────────────────────────────────────────────────────────
    lines    = raw.splitlines()
    filename = os.path.basename(rel_path)

    if len(lines) < config.BIG_FILE_LINE_LIMIT:
        return [_make_module(rel_path, raw)]

    # Large text files — split by type
    symbols = []

    if extension == ".json":
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                for key, value in data.items():
                    if not isinstance(value, (dict, list)):
                        continue
                    if isinstance(value, list):
                        value = value[:30]
                    src = json.dumps({key: value}, indent=2)
                    if len(src.splitlines()) > config.BIG_FILE_LINE_LIMIT:
                        src = "\n".join(src.splitlines()[:config.BIG_FILE_LINE_LIMIT]) + "\n... [truncated]"
                    symbols.append(Symbol(
                        id=f"{rel_path}::{filename}::{key}::1",
                        name=f"{filename}::{key}", kind="module",
                        file_path=rel_path, line_start=1, line_end=len(lines),
                        source_code=_cap(src),
                    ))
            elif isinstance(data, list):
                for i, chunk in enumerate([data[j:j+50] for j in range(0, len(data), 50)]):
                    symbols.append(Symbol(
                        id=f"{rel_path}::{filename}::chunk_{i}::1",
                        name=f"{filename}::chunk_{i}", kind="module",
                        file_path=rel_path, line_start=1, line_end=len(lines),
                        source_code=_cap(json.dumps(chunk, indent=2)),
                    ))
        except Exception:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            for i, chunk in enumerate(splitter.split_text(raw)):
                symbols.append(Symbol(
                    id=f"{rel_path}::chunk_{i}::1",
                    name=f"{filename}::chunk_{i}", kind="module",
                    file_path=rel_path, line_start=1, line_end=len(lines),
                    source_code=_cap(chunk),
                ))

    elif extension == ".md":
        splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=50)
        for i, chunk in enumerate(splitter.split_text(raw)):
            symbols.append(Symbol(
                id=f"{rel_path}::section_{i}::1",
                name=f"{filename}::section_{i}", kind="module",
                file_path=rel_path, line_start=1, line_end=len(lines),
                source_code=_cap(chunk),
            ))

    elif extension in (".yml", ".yaml"):
        try:
            data = yaml.safe_load(raw)
            if isinstance(data, dict):
                for key, value in data.items():
                    src = yaml.dump({key: value}, default_flow_style=False)
                    symbols.append(Symbol(
                        id=f"{rel_path}::{key}::1",
                        name=f"{filename}::{key}", kind="module",
                        file_path=rel_path, line_start=1, line_end=len(lines),
                        source_code=_cap(src),
                    ))
            else:
                return [_make_module(rel_path, raw)]
        except Exception:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            for i, chunk in enumerate(splitter.split_text(raw)):
                symbols.append(Symbol(
                    id=f"{rel_path}::chunk_{i}::1",
                    name=f"{filename}::chunk_{i}", kind="module",
                    file_path=rel_path, line_start=1, line_end=len(lines),
                    source_code=_cap(chunk),
                ))

    elif extension == ".html":
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        for i, chunk in enumerate(splitter.split_text(raw)):
            symbols.append(Symbol(
                id=f"{rel_path}::part_{i}::1",
                name=f"{filename}::part_{i}", kind="module",
                file_path=rel_path, line_start=1, line_end=len(lines),
                source_code=_cap(chunk),
            ))

    return symbols if symbols else [_make_module(rel_path, raw)]


# ── Public: parse_repo ────────────────────────────────────────────────────────
def parse_repo(repo_root: str) -> list:
    """Walk the repo and return all Symbol objects."""
    symbols = []
    repo_root = os.path.abspath(repo_root)
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in config.IGNORED_DIRS]
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in config.ALL_EXTENSIONS:
                continue
            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, repo_root)
            symbols.extend(extract_symbols(abs_path, rel_path, ext))
    return symbols


# ── Graph logic ───────────────────────────────────────────────────────────────
def build_graph(symbols: list) -> dict:
    """Build a function-call graph from symbols. Two-pass: nodes then edges."""
    graph = {"nodes": {}, "edges": {}}
    name_index: dict = {}   # name -> [symbol_id, ...]

    for sym in symbols:
        graph["nodes"][sym.id] = {
            "name":         sym.name,
            "kind":         sym.kind,
            "file_path":    sym.file_path,
            "line_start":   sym.line_start,
            "line_end":     sym.line_end,
            "parent_class": sym.parent_class,
        }
        graph["edges"][sym.id] = []
        name_index.setdefault(sym.name, []).append(sym.id)

    for sym in symbols:
        for called_name in sym.calls:
            for target_id in name_index.get(called_name, []):
                if target_id != sym.id:
                    graph["edges"][sym.id].append(target_id)
        graph["edges"][sym.id] = list(set(graph["edges"][sym.id]))

    return graph


def save_graph(graph: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)


def load_graph(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_neighbors(graph: dict, symbol_id: str, depth: int, max_per_node: int) -> list:
    """BFS from symbol_id up to `depth` hops; returns deduplicated neighbor IDs."""
    visited = {symbol_id}
    result  = []
    queue   = deque([(symbol_id, 0)])

    while queue:
        node_id, d = queue.popleft()
        if d >= depth:
            continue
        for neighbor in graph.get("edges", {}).get(node_id, [])[:max_per_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                result.append(neighbor)
                queue.append((neighbor, d + 1))

    return result
