# Environment Configuration
> Part of [AGENTS.md](../AGENTS.md) -- project guidance for AI coding agents.

## `.env` File

All model and clustering configuration is in `.env` at the project root.

### Model Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `ONNX_EMBEDDING_MODEL` | HuggingFace model identifier | `Xenova/all-MiniLM-L12-v2` |
| `ONNX_EMBEDDING_MODEL_PRECISION` | Model precision (`fp32`, `fp16`, `q8`) | `fp32` |
| `ONNX_EMBEDDING_MODEL_REQUIRES_PREFIX` | Whether model needs task prefixes | `false` |
| `ONNX_EMBEDDING_MODEL_DOCUMENT_PREFIX` | Prefix for training data embeddings | `search_document: ` |
| `ONNX_EMBEDDING_MODEL_QUERY_PREFIX` | Prefix for query embeddings | `search_query: ` |
| `ALLOW_REMOTE_MODELS` | Allow downloading models from HuggingFace | `true` |
| `LOCAL_MODEL_PATH` | Local model storage path | `models/` |
| `CACHE_DIR` | Downloaded model cache directory | `models/` |

### Supported Models

| Model | Precision | Size | Requires Prefix |
|-------|-----------|------|-----------------|
| `Xenova/all-MiniLM-L6-v2` | fp32, fp16, q8 | 90/45/23 MB | No |
| `Xenova/all-MiniLM-L12-v2` | fp32, fp16, q8 | 133/67/34 MB | No |
| `Xenova/paraphrase-multilingual-MiniLM-L12-v2` | fp32, fp16, q8 | 470/235/118 MB | No |
| `nomic-ai/modernbert-embed-base` | fp32, fp16, q8 | 568/284/146 MB | Yes (`search_document` / `search_query`) |

### Clustering Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_CLUSTERING` | Enable/disable clustering | `true` |
| `CLUSTERING_SIMILARITY_THRESHOLD` | Similarity threshold for cluster assignment (0-1) | `0.9` |
| `CLUSTERING_MIN_CLUSTER_SIZE` | Minimum phrases per cluster | `5` |
| `CLUSTERING_MAX_CLUSTERS` | Maximum clusters per topic | `5` |

Clustering settings can be overridden at runtime via `generate.js` CLI flags (see Development Commands).

## `labels-config.js`

Defines which topics to generate embeddings for and their match thresholds:

```js
export const labels = [
    { label: "disney", threshold: 0.4 },
    { label: "llamas", threshold: 0.4 },
    { label: "cookies", threshold: 0.4 },
]
```

- `label`: Must match a label value in `data/training_data.jsonl` (case-insensitive)
- `threshold`: Cosine similarity score that triggers a match (0-1). Lower = more permissive

## Adding New Topics

1. Add training phrases to `data/training_data.jsonl` with the new label
2. Add the label and threshold to `labels-config.js`
3. Run `node generate.js` to regenerate embeddings
