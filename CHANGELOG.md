# Changelog
All notable changes to this project will be documented in this file.


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