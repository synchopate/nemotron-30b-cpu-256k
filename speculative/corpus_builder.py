"""
RouteCrack Corpus Builder — Auto-generate corpus + cascade for any model.

Given a model (via Ollama or HuggingFace), generates training corpus,
builds hash cascade, and saves binary files ready for the C++ predictor.

Usage:
    from routecrack.corpus_builder import build_corpus
    build_corpus("Qwen/Qwen2.5-7B-Instruct", output_dir="/workspace")
"""
import struct
import time
import json
import os
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Optional


DIVERSE_PROMPTS = [
    # Python (30)
    "Write a Python function to check if a number is prime",
    "Implement a binary search in Python",
    "Write a Python class for a stack with push pop peek",
    "Create a Python function to reverse a string in place",
    "Write a Python decorator that measures execution time",
    "Implement a Python linked list with insert delete reverse",
    "Write a Python function to count character frequencies",
    "Create a Python generator for prime numbers",
    "Write a Python merge sort with explanation",
    "Implement a Python hash table with chaining",
    "Write a Python function for quicksort with partition",
    "Create a Python class for a binary search tree",
    "Write a Python function to validate parentheses",
    "Implement a Python set with add remove contains",
    "Write a Python function for extended Euclidean algorithm",
    "Create a Python class for a max-heap with heapify",
    "Write a Python function to rotate matrix 90 degrees",
    "Implement a Python deque with circular buffer",
    "Write a Python function to find all duplicates in O(n)",
    "Create a Python decorator for LRU caching",
    "Write a Python function for radix sort",
    "Create a Python context manager for timing blocks",
    "Write a Python function to find kth largest element",
    "Implement a Python ring buffer with overwrite",
    "Write a Python function for counting sort",
    "Create a Python class for weighted graph with Dijkstra",
    "Write a Python function to deep flatten any iterable",
    "Implement a Python fibonacci heap",
    "Write a Python function to convert any base to decimal",
    "Create a Python abstract factory pattern",
    # Web (20)
    "Write a FastAPI endpoint for user registration with validation",
    "Create a Flask REST API with pagination and filtering",
    "Write a Python WebSocket server with authentication",
    "Implement an HTTP client with connection pooling in Python",
    "Write a Django model with many-to-many relationships",
    "Create a FastAPI dependency injection system",
    "Write a Python async web crawler with rate limiting",
    "Implement JWT refresh token rotation in Python",
    "Write a Python function to parse HTTP headers",
    "Create a REST API versioning system in Python",
    "Write a Python CORS middleware from scratch",
    "Implement a server-sent events endpoint in Python",
    "Write a Python URL shortener with analytics",
    "Create a simple load balancer in Python",
    "Write a Python webhook delivery system",
    "Implement a Python API gateway with routing",
    "Write a Python function for request signing HMAC",
    "Create a Python streaming response handler",
    "Write a Python multipart file upload handler",
    "Implement a Python request rate limiter with sliding window",
    # Data (20)
    "Write a Python function to detect CSV delimiter",
    "Create a Python data pipeline with error recovery",
    "Write a Python function to merge DataFrames intelligently",
    "Implement a connection pool for PostgreSQL in Python",
    "Write a Python script for JSON schema migration",
    "Create a Python class for an in-memory column store",
    "Write a Python function for data type inference",
    "Implement a Python CDC change data capture system",
    "Write a Python function for exponential moving average",
    "Create a Python time series forecasting function",
    "Write a Python script for generating realistic test data",
    "Implement a SQL query builder with parameterization",
    "Write a Python function for Z-score outlier detection",
    "Create a Python data validation with custom rules",
    "Write a Python function to denormalize nested JSON",
    "Implement a key-value store with write-ahead log",
    "Write a Python function for stratified sampling",
    "Create a Python correlation matrix with p-values",
    "Write a Python query profiler and optimizer",
    "Implement a Python data deduplication with fuzzy matching",
    # DevOps (15)
    "Write a Dockerfile for Python FastAPI app multi-stage",
    "Create a GitHub Actions matrix build workflow",
    "Write a Python structured log parser",
    "Implement a health check with circuit breaker pattern",
    "Write a Kubernetes HPA configuration YAML",
    "Create a Makefile with dependency management for Python",
    "Write a Python resource usage alerting system",
    "Implement a configuration hot-reload system in Python",
    "Write a docker-compose with service mesh",
    "Create a Python zero-downtime deployment tool",
    "Write a Python nginx log analyzer",
    "Implement a service registry in Python",
    "Write a Python encrypted backup system",
    "Create a Python infrastructure drift detector",
    "Implement a Python request tracing system",
    # Algorithms (25)
    "Write a Python N-queens solver with optimization",
    "Implement Dijkstra with fibonacci heap in Python",
    "Write a Python Kahn topological sort",
    "Implement bidirectional A* in Python",
    "Write a Python Tarjan SCC algorithm",
    "Implement a Python red-black tree with delete",
    "Write a Python optimal matrix chain multiplication",
    "Create a Python generalized suffix tree",
    "Write a Python Hunt-Szymanski LCS algorithm",
    "Implement a Python persistent segment tree",
    "Write a Python Edmonds-Karp max flow",
    "Create a Python Aho-Corasick string matcher",
    "Write a Python Graham scan convex hull",
    "Implement a Python B+ tree with range queries",
    "Write a Python consistent hashing with vnodes",
    "Create a Python counting bloom filter",
    "Implement a Python skip list with search",
    "Write a Python Huffman coding compressor",
    "Implement a Python rope data structure for text",
    "Write a Python k-d tree with nearest neighbor",
    "Create a Python Burrows-Wheeler transform",
    "Write a Python Raft consensus simplified",
    "Implement a Python treap with implicit keys",
    "Write a Python suffix automaton",
    "Create a Python palindromic tree",
    # Agentic (20)
    "Write a Python JSON schema validator returning structured errors",
    "Create a Python AST function signature extractor",
    "Write a Python subprocess executor with streaming and timeout",
    "Implement a Python retry with exponential backoff and jitter",
    "Write a Python unified diff parser into structured changes",
    "Create a Python import dependency graph generator",
    "Write a Python cyclomatic complexity calculator",
    "Implement a Python code formatter with diff output",
    "Write a Python multi-file regex search and replace",
    "Create a Python docstring to OpenAPI converter",
    "Write a Python deep JSON diff with path reporting",
    "Implement a Python YAML/TOML config validator",
    "Write a Python log anomaly detector with statistics",
    "Create a Python inotify file watcher with debounce",
    "Write a Python test fixture generator from SQL schema",
    "Implement a Python function profiler with flame graph data",
    "Write a Python cron expression parser and scheduler",
    "Create a Python conventional commit changelog generator",
    "Write a Python API mock generator from OpenAPI spec",
    "Implement a Python dead code detector using AST",
    # Extra patterns (20)
    "Write a Python function for run-length encoding and decoding",
    "Implement a Python LRU cache with O(1) operations",
    "Write a Python Sudoku solver with constraint propagation",
    "Create a Python thread pool executor from scratch",
    "Write a Python function to check if graph is bipartite",
    "Implement a Python trie with autocomplete functionality",
    "Write a Python function for reservoir sampling",
    "Create a Python rate limiter using token bucket algorithm",
    "Write a Python async task queue with worker pool",
    "Implement a Python producer consumer with bounded queue",
    "Write a Python function to detect cycle in directed graph",
    "Create a Python class for polynomial operations",
    "Write a Python function for power set generation",
    "Implement a Python semaphore from scratch",
    "Write a Python function to find articulation points",
    "Create a Python observable pattern implementation",
    "Write a Python function for coin change problem dynamic",
    "Implement a Python read-write lock",
    "Write a Python function to find bridges in a graph",
    "Create a Python immutable persistent data structure",
]


def fnv64(tokens):
    h = 14695981039346656037
    for t in tokens:
        h ^= (int(t) & 0xFFFFFFFF)
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def generate_via_ollama(model, prompt, max_tok=300, ollama_url="http://localhost:11434"):
    """Generate text via Ollama API."""
    import requests, re
    r = requests.post(f"{ollama_url}/api/generate", json={
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "num_predict": max_tok}
    }, timeout=300)
    data = r.json()
    response = data.get("response", "")
    # Qwen3.5 thinking mode: response is empty, content in "thinking"
    if not response and data.get("thinking"):
        response = data["thinking"]
    # Also check if response contains think tags
    if not response:
        # Some models return empty response with all content in thinking
        response = data.get("thinking", "")
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    # If still empty after stripping think tags, use raw thinking
    if not response and data.get("thinking"):
        response = data["thinking"][:2000]  # cap at 2K chars
    return response


def generate_via_sglang(model, prompt, max_tok=300, port=30000):
    """Generate text via SGLang/OpenAI-compatible API."""
    import requests
    r = requests.post(f"http://localhost:{port}/v1/chat/completions", json={
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tok, "temperature": 0
    }, timeout=180)
    return r.json()["choices"][0]["message"]["content"]


def build_corpus(
    model: str,
    output_dir: str = ".",
    n_prompts: int = 150,
    max_tokens: int = 300,
    backend: str = "auto",
    ollama_url: str = "http://localhost:11434",
    sglang_port: int = 30000,
    tokenizer_name: Optional[str] = None,
):
    """Generate corpus from model and build cascade + binary files.

    Args:
        model: Model name (Ollama or HuggingFace format)
        output_dir: Where to save cascade + corpus files
        n_prompts: Number of training prompts
        max_tokens: Max tokens per response
        backend: "ollama", "sglang", or "auto"
        tokenizer_name: HuggingFace tokenizer (auto-detected from model name)

    Returns:
        dict with paths to generated files and stats
    """
    from transformers import AutoTokenizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = model.replace("/", "_").replace("-", "_").replace(":", "_")

    # Auto-detect backend
    if backend == "auto":
        import requests
        try:
            requests.get(f"http://localhost:{sglang_port}/health", timeout=2)
            backend = "sglang"
        except:
            try:
                requests.get(f"{ollama_url}/api/tags", timeout=2)
                backend = "ollama"
            except:
                raise RuntimeError("No backend found. Start Ollama or SGLang first.")

    # Auto-detect tokenizer
    if tokenizer_name is None:
        tok_map = {
            "deepseek": "deepseek-ai/DeepSeek-V2-Lite",
            "llama": "meta-llama/Llama-3.1-8B-Instruct",
            "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
            "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        }
        tokenizer_name = model
        for key, val in tok_map.items():
            if key in model.lower():
                tokenizer_name = val
                break

    print(f"[RouteCrack] Building corpus for {model}")
    print(f"  Backend: {backend}")
    print(f"  Tokenizer: {tokenizer_name}")
    print(f"  Prompts: {n_prompts}")

    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if backend == "sglang":
        def gen_fn(p):
            return generate_via_sglang(model, p, max_tokens, sglang_port)
    else:
        def gen_fn(p):
            return generate_via_ollama(model, p, max_tokens, ollama_url)

    prompts_used = DIVERSE_PROMPTS[:n_prompts]
    all_tokens = []

    for i, p in enumerate(prompts_used):
        try:
            text = gen_fn(p)
            pt = tok.encode(p, add_special_tokens=False)
            rt = tok.encode(text, add_special_tokens=False)
            all_tokens.extend(pt + rt)
        except Exception as e:
            print(f"  ERROR prompt {i}: {e}")
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{n_prompts}: {len(all_tokens):,} tokens", flush=True)

    print(f"  Total: {len(all_tokens):,} tokens")

    # Build cascade (top-4 candidates per context)
    CASCADE_TOPK = 4
    files = {}
    for k in [3, 4, 5, 6, 8, 10]:
        raw = defaultdict(Counter)
        for j in range(k, len(all_tokens)):
            raw[tuple(all_tokens[j-k:j])][all_tokens[j]] += 1
        entries = []
        for ctx, cts in raw.items():
            total = sum(cts.values())
            if total < 2: continue
            top = cts.most_common(CASCADE_TOPK)
            preds = [t for t, _ in top] + [0] * (CASCADE_TOPK - len(top))
            confs = [c * 100 // total for _, c in top] + [0] * (CASCADE_TOPK - len(top))
            entries.append((fnv64(list(ctx)), preds, confs))
        entries.sort(key=lambda e: e[0])
        fpath = output_dir / f"hc_{safe_name}_k{k}.bin"
        with open(fpath, "wb") as f:
            # v2 header: n_entries, k, n_candidates, reserved
            f.write(struct.pack("<IIII", len(entries), k, CASCADE_TOPK, 0))
            for h, preds, confs in entries:
                f.write(struct.pack("<Q", h))
                for p in preds:
                    f.write(struct.pack("<i", p))
                for c in confs:
                    f.write(struct.pack("<i", c))
        files[f"k{k}"] = str(fpath)
        print(f"  k={k}: {len(entries):,} entries (top-{CASCADE_TOPK}) → {fpath.name}")

    # Save corpus binary
    corpus_path = output_dir / f"corpus_{safe_name}.bin"
    with open(corpus_path, "wb") as f:
        for t in all_tokens:
            f.write(struct.pack("<i", t))
    files["corpus"] = str(corpus_path)

    # Save metadata
    meta = {
        "model": model, "tokenizer": tokenizer_name,
        "n_prompts": n_prompts, "n_tokens": len(all_tokens),
        "files": files, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = output_dir / f"corpus_{safe_name}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Metadata: {meta_path.name}")
    print(f"[RouteCrack] Corpus built: {len(all_tokens):,} tokens, "
          f"{sum(1 for k in files if k.startswith('k'))} cascade tables")
    return meta
