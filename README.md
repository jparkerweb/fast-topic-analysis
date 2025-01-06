# 🏷️ Fast Topic Analysis

A tool for analyzing text against predefined topics using average weight embeddings and cosine similarity.

![Fast Topic Analysis](./img/buckets.jpg)

#### Maintained by
<a href="https://www.equilllabs.com">
  <img src="https://raw.githubusercontent.com/jparkerweb/eQuill-Labs/refs/heads/main/src/static/images/logo-text-outline.png" alt="eQuill Labs" height="30">
</a>

<br>
<br>

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

- Change the model settings in `.env` to use different embedding models and configurations:
  ```env
  # Model and precision
  ONNX_EMBEDDING_MODEL="Xenova/all-MiniLM-L12-v2"
  ONNX_EMBEDDING_MODEL_PRECISION=fp32

  # Available Models and their configurations:
  # | Model                                        | Precision      | Size                   | Requires Prefix | Data Prefix     | Search Prefix |
  # | -------------------------------------------- | -------------- | ---------------------- | --------------- | --------------- | ------------- |
  # | Xenova/all-MiniLM-L6-v2                      | fp32, fp16, q8 | 90 MB, 45 MB, 23 MB    | false           | null            | null          |
  # | Xenova/all-MiniLM-L12-v2                     | fp32, fp16, q8 | 133 MB, 67 MB, 34 MB   | false           | null            | null          |
  # | Xenova/paraphrase-multilingual-MiniLM-L12-v2 | fp32, fp16, q8 | 470 MB, 235 MB, 118 MB | false           | null            | null          |
  # | nomic-ai/modernbert-embed-base               | fp32, fp16, q8 | 568 MB, 284 MB, 146 MB | true            | search_document | search_query  |
  ```
- Change the thresholds defined in `labels-config.js` per topic to change the similarity score that triggers a match.
- Add more test messages to the `test-messages` directory to test against.
- Add more training data to `data/training_data.jsonl` to improve the topic embeddings.

### Task Instruction Prefixes

Some models require specific prefixes to optimize their performance for different tasks. When a model has `Requires Prefix: true`, you must use the appropriate prefix:

- `Data Prefix`: Used when generating embeddings from training data
- `Search Prefix`: Used when generating embeddings for search/query text

For example, `nomic-ai/modernbert-embed-base` requires:
- `search_document` prefix for training data
- `search_query` prefix for search queries

Models with `Requires Prefix: false` will ignore any prefix settings.

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

## 📺 Video Demo
[![video](https://img.youtube.com/vi/SsPKA2Sy1pE/0.jpg)](https://www.youtube.com/watch?v=SsPKA2Sy1pE)
