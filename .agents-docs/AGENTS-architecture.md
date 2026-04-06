# Architecture
> Part of [AGENTS.md](../AGENTS.md) -- project guidance for AI coding agents.

## Two-Phase Pipeline

### Phase 1: Embedding Generation (`generate.js`)
1. Reads training data from `data/training_data.jsonl` (JSONL format: `{"text": "...", "label": "..."}`)
2. Filters phrases by topic labels defined in `labels-config.js`
3. Generates embeddings for each phrase using `modules/embedding.js` (thin wrapper around `embedding-utils`)
4. Clusters embeddings per topic using `clusterEmbeddings()` or `hdbscan()` from `embedding-utils`
5. Saves each cluster as a separate JSON file in `data/topic_embeddings/` (format: `<topic>-cluster-<N>-of-<M>.json`)
6. Creates a manifest file (`data/incremental-manifest.json`) for subsequent incremental updates

#### Incremental Mode (`generate.js --incremental`)
1. Loads and validates the manifest (content hash + model/precision check)
2. Detects new lines appended to `training_data.jsonl`
3. Embeds only the new phrases
4. Assigns each new embedding to the nearest existing cluster using `assignToCluster()` from `embedding-utils`
5. Updates cluster centroids using `batchIncrementalAverage` from `embedding-utils`
6. Updates the manifest

### Phase 2: Analysis (`run-demo.js`)
1. Loads all pre-generated topic embedding JSON files
2. Parses input text into sentences using `sentence-parse`
3. Generates embeddings for each sentence
4. Compares each sentence embedding against all topic clusters using cosine similarity
5. Reports matches where similarity exceeds the topic's threshold

## Core Modules (`modules/`)

| Module | Exports | Purpose |
|--------|---------|---------|
| `embedding.js` | `provider`, `prefixConfig`, `generateEmbeddings()`, `combineTopicEmbeddings()` | Thin wrapper around `embedding-utils`'s `createLocalProvider()`. Initializes the ONNX embedding provider, generates embeddings with optional prefix support |
| `manifest.js` | `createManifest()`, `loadManifest()`, `validateManifest()`, `getNewLines()`, `updateManifest()` | Manifest CRUD for incremental mode: tracks processed line count, SHA-256 content hash, model/precision info |
| `utils.js` | `toBoolean()` | String-to-boolean conversion for env var parsing |

## Key Configuration Files

| File | Purpose |
|------|---------|
| `labels-config.js` | Defines topic labels and their similarity thresholds (e.g., `{ label: "disney", threshold: 0.4 }`) |
| `.env` | Model selection, precision, prefix config, clustering params, cache paths |
| `data/training_data.jsonl` | Training phrases with topic labels |
| `data/incremental-manifest.json` | Incremental processing state (line count, content hash, model info) |

## Clustering Algorithms

Two algorithms are available via `--algorithm` flag or `CLUSTERING_ALGORITHM` env var:

### Default (Agglomerative)
Provided by `embedding-utils`'s `clusterEmbeddings()`. Config (threshold, minClusterSize, maxClusters) is passed as arguments. Use `getPreset(name)` for preset handling. The algorithm:
1. Initialize first cluster with first embedding
2. For each remaining embedding: assign to most similar cluster if above threshold, otherwise create new cluster (up to `maxClusters`)
3. Clusters smaller than `minClusterSize` are redistributed to the nearest valid cluster, or combined into a single "miscellaneous" cluster if no valid clusters exist
4. Final output: array of `{ centroid, members, size, cohesion, labels }` objects

### HDBSCAN
Provided by `embedding-utils`'s `hdbscan()`. Density-based clustering that auto-determines cluster count. Uses `minClusterSize` parameter. Noise points are reassigned to nearest cluster via `assignToCluster()`. Falls back to single cluster if no clusters found. More conservative — works best with larger datasets.

### Quality Metrics
- **Cohesion** (per-cluster): average member-to-centroid cosine similarity, via `centroidCohesion()`
- **Silhouette score** (global): cluster separation quality (-1 to +1), via `silhouetteScore()`. Returns 0 for single-cluster topics.

## Dependencies

- `embedding-utils` (^0.3.0) -- Vector math, clustering (agglomerative + HDBSCAN), cosine similarity, silhouette score, embedding provider
- `@huggingface/transformers` -- ONNX model loading and inference (local, no API)
- `sentence-parse` -- Text-to-sentence splitting
- `chalk` -- Terminal color output
- `dotenv` -- Environment variable loading

## Module System

The project uses **ES modules** (`"type": "module"` in package.json). All imports use `import`/`export` syntax. The `embedding.js` module uses top-level `await` for pipeline initialization.
