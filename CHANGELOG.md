# Changelog
All notable changes to this project will be documented in this file.

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