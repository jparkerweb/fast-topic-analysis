# AGENTS.md

This file provides guidance to AI coding agents like Claude Code (claude.ai/code), Cursor AI, Codex, Gemini CLI, GitHub Copilot, and other AI coding assistants when working with code in this repository.

## Project Overview

Fast Topic Analysis is a Node.js (ES modules) tool for identifying topic matches in text using embedding-based semantic analysis with clustering. It generates weighted average embeddings per topic from training data, clusters them for nuanced variation detection, and compares input text against those clusters using cosine similarity. The project uses ONNX models via `@huggingface/transformers` for local embedding generation -- no external API calls.

Current version: **1.2.0** | License: ISC | Runtime: Node.js

## Development Commands

Common commands for building, testing, and running the project. Includes npm scripts, CLI flags for both `generate.js` and `run-demo.js`, and clustering preset options.

Details: [Development Commands](./.agents-docs/AGENTS-development-commands.md)

## Architecture

Two-phase pipeline: (1) `generate.js` creates clustered topic embeddings from training data, (2) `run-demo.js` analyzes text against those embeddings. Core modules in `modules/` handle embedding generation, cosine similarity, and clustering logic. Configuration lives in `.env` and `labels-config.js`.

Details: [Architecture](./.agents-docs/AGENTS-architecture.md)

## Environment Configuration

Model selection, precision, prefix settings, clustering parameters, and local model caching are all configured via `.env`. Topic labels and thresholds are defined in `labels-config.js`. Clustering behavior can be overridden at runtime via CLI flags or presets.

Details: [Environment Configuration](./.agents-docs/AGENTS-environment-configuration.md)

## How to Use This File

The sections above contain brief summaries. Agents should follow the markdown links to `.agents-docs/` for full details -- only read what is relevant to the current task. Inline sections (Project Overview, Git Commit Messages) contain all necessary information directly. For Development Commands, Architecture, and Environment Configuration, click through to the detail files for complete reference.
