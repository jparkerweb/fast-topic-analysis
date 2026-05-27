# Changelog
All notable changes to this project will be documented in this file.


## [Unreleased]
### Updated
- Upgraded `embedding-utils` from `^0.3.0` to `^0.4.0`
- Prefixes and pooling are now auto-detected from `embedding-utils`'s built-in model registry. Removed `ONNX_EMBEDDING_MODEL_REQUIRES_PREFIX`. `ONNX_EMBEDDING_MODEL_DOCUMENT_PREFIX` and `ONNX_EMBEDDING_MODEL_QUERY_PREFIX` remain as optional overrides.
- `generateEmbeddings()` now accepts `inputType: 'document' | 'query'` instead of manual `prefix` strings
- Fixed incorrect model IDs in `.env`: `nomic-ai/modernbert-embed-base` → `nomic-ai/nomic-embed-text-v1.5`, `BAAI/bge-small-en-v1.5` → `Xenova/bge-small-en-v1.5`, `Xenova/all-distilroberta-v1` → `Xenova/distilroberta-base`
- Expanded `.env` model registry from 6 to all 28 supported models with dimensions, max tokens, pooling method, and prefix info
- Renamed `test/v030-features-test.js` to `test/embedding-utils-features-test.js` for version-agnostic naming

## [1.5.0] - 2026-04-06
### Added
- HDBSCAN clustering algorithm as alternative to default agglomerative clustering (`--algorithm hdbscan` CLI flag, `CLUSTERING_ALGORITHM` env var)
- Silhouette score as global clustering quality metric (displayed in console output and saved to cluster JSON files)
- `assignToCluster()` from `embedding-utils` for cleaner incremental cluster assignment
- New test suite for v0.3.0 features: `test/v030-features-test.js` (16 tests covering assignToCluster, silhouetteScore, HDBSCAN, Float32Array handling)

### Updated
- Upgraded `embedding-utils` from `^0.2.0` to `^0.3.0`
- Float32Array compatibility: all embedding serialization now uses `Array.from()` to handle v0.3.0's typed array returns
- Incremental mode refactored to use `assignToCluster()` instead of manual cosine similarity loop
- HDBSCAN noise points are automatically reassigned to nearest cluster; falls back to single cluster if HDBSCAN finds no clusters
- Cleaned up redundant `calculateCohesion` test helper in favor of direct `centroidCohesion()` calls

## [1.4.0] - 2026-04-03
### Updated
- Migrated from custom embedding/similarity/clustering modules to `embedding-utils` npm package
- Replaced `modules/similarity.js` and `modules/clusterEmbeddings.js` with `embedding-utils` imports
- Rewrote `modules/embedding.js` as thin wrapper around `embedding-utils`'s `createLocalProvider()`
- Updated `@huggingface/transformers` from `^3.8.1` to `^4.0.1`
- Updated project documentation (AGENTS.md, architecture docs)

### Removed
- `modules/similarity.js` (replaced by `cosineSimilarity` from `embedding-utils`)
- `modules/clusterEmbeddings.js` (replaced by library clustering functions)
- `test/demo-clustering.js` and `test/weighted-average-test.js` (obsolete standalone tests)

## [1.3.0] - 2026-03-27
### Added
- Incremental embedding updates via `node generate.js --incremental`
- Manifest module (`modules/manifest.js`) for tracking processing state and data integrity
- SHA-256 content hashing to detect modified training data
- Model/precision mismatch detection between runs
- Weighted average centroid updates without re-embedding existing data
- Unit tests for weighted average math, manifest validation, and getNewLines
- Integration test comparing full generation vs incremental generation
- Edge case tests for missing manifest, model mismatch, and no new data scenarios

## [1.2.0] - 2025-02-25
### ✨ Added
- Implemented embedding clustering to create multiple embeddings per topic
- Added cohesion score calculation for evaluating cluster quality
- Created configuration presets (high-precision, balanced, performance, legacy)
- Added command-line arguments for customizing clustering behavior
- Implemented unit tests for clustering algorithm
- Updated README with comprehensive clustering explanation

### 📦 Updated
- Renamed run-test.js to run-demo.js for clarity
- Improved file organization with dedicated test directory
- Enhanced output to display cluster information and cohesion scores
- Updated package.json description to better reflect project functionality

## [1.1.0] - 2025-01-06
### ✨ Added
- Support for `task instruction prefixes` (defined in `.env` file)
  see this [nomic embedding model card for reference](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5#task-instruction-prefixes)
- Noic's new `modernbert-embed-base` model has been added to the `.env` comments as an example

## [1.0.2] - 2024-12-18
### 📦 Updated
- Changed sentence parsing library to `sentence-parse`

## [1.0.1] - 2024-12-15
### 📦 Updated
- Updated TransformersJS to 3.2.0

## [1.0.0] - 2024-02-21
### ✨ Added
- Topic embedding generation from training data
- Interactive test runner for analyzing text
- Support for custom embedding models via ONNX
- Configurable similarity thresholds per topic
- Automatic model caching
- Verbose and minimal logging modes
- Configuration persistence
- Training data support in JSONL format