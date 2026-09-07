# Hybrid Search Engine (BM25 + HNSW) in Rust

A search engine core built from scratch in Rust, combining traditional
keyword search (BM25) with AI-powered semantic vector search (HNSW),
merged into a single ranked result via Reciprocal Rank Fusion — the same
general approach used by modern production search systems.

Built as a systems/ML-infra portfolio project to demonstrate implementing
non-trivial algorithms from research papers, rather than relying on
pre-built vector database wrappers.

## What it does

Given a search query, the engine:
1. Finds keyword matches using a **BM25 inverted index** (built from scratch)
2. Finds semantically similar documents using an **HNSW graph** (built from
   scratch) over real AI-generated sentence embeddings
3. Merges both ranked lists into one result using **Reciprocal Rank Fusion**

This means a query like `"affordable notebook computer"` correctly
surfaces a document about a `"budget laptop... cheap computer"` — the
exact words don't overlap, but the AI embedding captures the shared
meaning.

## Architecture

| Crate | Responsibility |
|---|---|
| `crates/bm25` | Keyword inverted index + BM25 ranking, from scratch |
| `crates/hnsw` | Hierarchical Navigable Small World graph for approximate nearest-neighbor vector search, from scratch, SIMD-accelerated |
| `crates/embeddings` | Wraps ONNX Runtime to run a real pretrained sentence-transformer model (`all-MiniLM-L6-v2`) for generating embeddings |
| `crates/fusion` | Combines BM25 + HNSW rankings via Reciprocal Rank Fusion; also handles saving/loading the index to disk via memory-mapped files |
| `crates/server` | gRPC network service exposing Search/AddDocument over the network (via `tonic`) |
| `crates/cli` | Runnable demo binaries |

## Results

- **HNSW vs. brute-force vector search:** ~20-25x faster at ~100% recall@10
  on 50,000 random vectors (see `crates/hnsw/examples/benchmark.rs`)
- **SIMD-accelerated distance calculation:** ~5-7x faster than the
  scalar (one-number-at-a-time) version, with bit-identical correctness
  (see `crates/hnsw/examples/simd_benchmark.rs`)
- **Persistence:** index save/load via `mmap` produces identical search
  results to the original in-memory index, with no full-file read required
  before search is possible

## Running it

Requires:
- [Rust](https://rustup.rs) (stable)
- An ONNX-exported sentence-transformer model + tokenizer, e.g.
  [`Xenova/all-MiniLM-L6-v2`](https://huggingface.co/Xenova/all-MiniLM-L6-v2)
  — download `onnx/model.onnx` and `tokenizer.json` into a local `model/` folder
- The [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases) shared
  library for your platform (tested against v1.18.0)
- A `protoc` binary (for the gRPC server) — a prebuilt binary from the
  [protobuf releases](https://github.com/protocolbuffers/protobuf/releases)
  works fine, no build tools required

```bash
# Run all tests
cargo test --workspace

# BM25 keyword search demo
cargo run -p search-cli --bin search-cli

# Full hybrid search demo (needs model/model.onnx + model/tokenizer.json)
set ORT_DYLIB_PATH=path\to\onnxruntime.dll   # Windows
cargo run -p search-cli --bin hybrid_demo

# Save/load persistence demo (no AI model needed)
cargo run -p search-cli --bin persistence_demo

# Interactive search -- type your own queries and get real results
set ORT_DYLIB_PATH=path\to\onnxruntime.dll   # Windows
cargo run -p search-cli --bin interactive_search

# HNSW speed benchmarks
cargo run -p hnsw --release --example benchmark
cargo run -p hnsw --release --example simd_benchmark

# gRPC server + client (two terminals)
set PROTOC=path\to\protoc.exe
set ORT_DYLIB_PATH=path\to\onnxruntime.dll
cargo run -p server --bin server        # terminal 1
cargo run -p server --bin client_demo   # terminal 2
```

## Why these design choices

- **BM25 before HNSW:** BM25 is exact and deterministic, making it easy to
  unit-test correctness before tackling HNSW's more complex, approximate
  algorithm.
- **Squared Euclidean distance internally:** avoids the sqrt call on the
  hot path — sqrt is monotonic, so it never changes which candidate is
  closer, only its exact reported distance.
- **RRF over score normalization:** BM25 scores and vector distances live
  on incomparable scales; RRF only needs each engine's rank order, sidestepping
  that problem entirely.
- **mmap over plain file reads:** lets the OS page a large index in lazily
  from disk instead of requiring a full up-front read before any query can
  run.
