# Development Commands
> Part of [AGENTS.md](../AGENTS.md) -- project guidance for AI coding agents.

## Install Dependencies

```bash
npm install
```

First run will download the ONNX embedding model to the `models/` directory (configured via `.env`). Subsequent runs use the cached model.

## Generate Topic Embeddings

```bash
npm run generate
# or
node generate.js
```

Cleans `data/topic_embeddings/`, processes `data/training_data.jsonl`, clusters embeddings per topic, and saves results as JSON files.

### Clustering Presets

```bash
node generate.js --preset high-precision   # threshold=0.95, min=3, max=8
node generate.js --preset balanced          # threshold=0.9, min=5, max=5 (default)
node generate.js --preset performance       # threshold=0.85, min=10, max=3
node generate.js --preset legacy            # disables clustering entirely
```

### Individual Clustering Flags

```bash
node generate.js --enable-clustering <bool>
node generate.js --similarity-threshold <0-1>
node generate.js --min-cluster-size <num>
node generate.js --max-clusters <num>
```

## Run Analysis Demo

```bash
npm start
# or
node run-demo.js [options] [file]
```

Interactive demo that analyzes test messages against generated topic embeddings.

| Flag | Description |
|------|-------------|
| `--verbose, -v` | Enable verbose logging |
| `--quiet, -q` | Disable verbose logging |
| `--show-matches, -s` | Show matched sentences |
| `--hide-matches, -h` | Hide matched sentences |
| `[file]` | Test message number (1-N) or filename |

Examples:
```bash
node run-demo.js 2
node run-demo.js message-1.txt
node run-demo.js --quiet --show-matches 3
```

Preferences are saved to `run-demo-config.json` between runs.

## Run Tests

```bash
npm test
# or
node test/cluster-test.js
```

Tests cover: basic clustering, clustering with phrases, disabled clustering, small cluster merging, average embedding calculation, and cohesion scoring. Tests use Node's built-in `assert` module (no test framework).

## Clean Install

```bash
npm run clean
```

Removes `node_modules/` and `package-lock.json`, then reinstalls.
