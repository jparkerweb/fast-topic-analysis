# 🏷️ Fast Topic Analysis

A tool for analyzing text against predefined topics using embeddings and cosine similarity.

![Fast Topic Analysis](./img/buckets.jpg)

## Overview

This project consists of two main components:
1. A generator (`generate.js`) that creates topic embeddings from training data
2. A test runner (`run-test.js`) that analyzes text against these topic embeddings

## Setup

Install dependencies:

```bash
npm install
```

## Usage

### Generating Topic Embeddings

```bash
node generate.js
```

This will:
- Clean the `data/topic_embeddings` directory
- Process training data from `data/training_data.jsonl`
- Generate embeddings for each topic defined in `labels-config.js`
- Save embeddings as JSON files in `data/topic_embeddings/`

### Running Analysis

```bash
node run-test.js
```

The test runner provides an interactive interface to:
1. Choose logging verbosity
2. Optionally show matched sentences if verbose logging is disabled
3. Select a test message file to analyze

Configuration preferences (last used file, verbosity, etc.) are automatically saved in `run-test-config.json`.

#### 🚨 First Run Model Download

The first time a model is used (e.g. `generate.js` or `run-test.js`), it will be downloaded and cached to the directory speciifed in `.env`. All subsequent runs will be fast as the model will be loaded from the cache.


### Output

The analysis will show:
- Similarity scores between the test text and each topic
- Execution time
- Total comparisons made
- Number of matches found
- Model information

## File Structure

```
├── data/
│   ├── training_data.jsonl          # Training data
│   └── topic_embeddings/            # Generated embeddings
├── test-messages/                   # Test files
├── modules/
│   ├── embedding.js                 # Embedding functions
│   └── similarity.js                # Similarity calculation
├── generate.js                      # Embedding generator
├── run-test.js                      # Test runner
└── labels-config.js                 # Topic definitions
```

## Customizing

- Change the `ONNX_EMBEDDING_MODEL` and `ONNX_EMBEDDING_MODEL_PRECISION` in `.env` to use a different embedding model and precision.
- Change the thresholds defined in `labels-config.js` per topic to change the similarity score that triggers a match.
- Add more test messages to the `test-messages` directory to test against.
- Add more training data to `data/training_data.jsonl` to improve the topic embeddings.

### Training Data

The training data is a JSONL file that contains the training data. Each line is a JSON object with the following fields:
- `text`: The text to be analyzed
- `label`: The label of the topic

```jsonl
{"text": "amphibians, croaks, wetlands, camouflage, metamorphosis", "label": "frogs"}
{"text": "jumping, ponds, tadpoles, moist skin, diverse habitats", "label": "frogs"}
{"text": "waterfowl, quacking, ponds, waddling, migration", "label": "ducks"}
{"text": "feathers, webbed feet, lakes, nesting, foraging", "label": "ducks"}
{"text": "dabbling, flocks, wetlands, bills, swimming", "label": "ducks"}
```

The training data is used to generate the topic embeddings. The more training data you have, the better the topic embeddings will be.
The labels to be used when generating the topic embeddings are defined in `labels-config.js`.
