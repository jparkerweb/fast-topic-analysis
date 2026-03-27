# Architecture
> Part of [AGENTS.md](../AGENTS.md) -- project guidance for AI coding agents.

## Two-Phase Pipeline

### Phase 1: Embedding Generation (`generate.js`)
1. Reads training data from `data/training_data.jsonl` (JSONL format: `{"text": "...", "label": "..."}`)
2. Filters phrases by topic labels defined in `labels-config.js`
3. Generates embeddings for each phrase using the ONNX model via `modules/embedding.js`
4. Clusters embeddings per topic using `modules/clusterEmbeddings.js`
5. Saves each cluster as a separate JSON file in `data/topic_embeddings/` (format: `<topic>-cluster-<N>-of-<M>.json`)

### Phase 2: Analysis (`run-demo.js`)
1. Loads all pre-generated topic embedding JSON files
2. Parses input text into sentences using `sentence-parse`
3. Generates embeddings for each sentence
4. Compares each sentence embedding against all topic clusters using cosine similarity
5. Reports matches where similarity exceeds the topic's threshold

## Core Modules (`modules/`)

| Module | Exports | Purpose |
|--------|---------|---------|
| `embedding.js` | `generateEmbeddings()`, `combineTopicEmbeddings()`, `prefixConfig` | Initializes the HuggingFace ONNX pipeline, generates embeddings with optional prefix support, handles weighted averaging |
| `similarity.js` | `cosineSimilarity()` | Pure cosine similarity calculation between two vectors |
| `clusterEmbeddings.js` | `clusterEmbeddings()`, `calculateAverageEmbedding()`, `updateClusteringConfig()`, `clusteringConfig` | Similarity-based clustering algorithm with configurable thresholds, min sizes, and max clusters. Handles small cluster redistribution |
| `utils.js` | `toBoolean()` | String-to-boolean conversion for env var parsing |

## Key Configuration Files

| File | Purpose |
|------|---------|
| `labels-config.js` | Defines topic labels and their similarity thresholds (e.g., `{ label: "disney", threshold: 0.4 }`) |
| `.env` | Model selection, precision, prefix config, clustering params, cache paths |
| `data/training_data.jsonl` | Training phrases with topic labels |

## Clustering Algorithm

The clustering in `clusterEmbeddings.js` works as follows:
1. Initialize first cluster with first embedding
2. For each remaining embedding: assign to most similar cluster if above threshold, otherwise create new cluster (up to `maxClusters`)
3. Clusters smaller than `minClusterSize` are redistributed to the nearest valid cluster, or combined into a single "miscellaneous" cluster if no valid clusters exist
4. Final output: array of `{ embeddings, phrases, centroid }` objects

## Dependencies

- `@huggingface/transformers` -- ONNX model loading and inference (local, no API)
- `sentence-parse` -- Text-to-sentence splitting
- `chalk` -- Terminal color output
- `dotenv` -- Environment variable loading

## Module System

The project uses **ES modules** (`"type": "module"` in package.json). All imports use `import`/`export` syntax. The `embedding.js` module uses top-level `await` for pipeline initialization.
