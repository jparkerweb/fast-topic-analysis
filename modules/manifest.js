import fs from 'fs';
import crypto from 'crypto';

/**
 * Parse JSONL file and return valid parsed lines.
 * Matches the filtering logic in generate.js (lines 44-52):
 * JSON.parse each line, filter out nulls.
 */
function parseJsonlLines(jsonlPath) {
  const raw = fs.readFileSync(jsonlPath, 'utf8');
  return raw.split('\n')
    .map(line => {
      try {
        return JSON.parse(line);
      } catch (e) {
        return null;
      }
    })
    .filter(item => item !== null);
}

/**
 * Compute SHA-256 hash of valid JSONL lines.
 * Lines are stringified and joined by newline to ensure deterministic hashing.
 */
function computeHash(validLines) {
  const content = validLines.map(line => JSON.stringify(line)).join('\n');
  return 'sha256:' + crypto.createHash('sha256').update(content).digest('hex');
}

/**
 * Create a new manifest after a full generation run.
 */
export function createManifest(jsonlPath, manifestPath, model, precision) {
  const validLines = parseJsonlLines(jsonlPath);
  const manifest = {
    lastProcessedLine: validLines.length,
    contentHash: computeHash(validLines),
    embeddingModel: model,
    modelPrecision: precision,
    lastRunDate: new Date().toISOString()
  };
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  return manifest;
}

/**
 * Load an existing manifest file. Returns null if it doesn't exist.
 */
export function loadManifest(manifestPath) {
  if (!fs.existsSync(manifestPath)) {
    return null;
  }
  return JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
}

/**
 * Validate a manifest against the current JSONL file and environment.
 * Returns { valid: boolean, reason: string|null }.
 */
export function validateManifest(manifest, jsonlPath) {
  const validLines = parseJsonlLines(jsonlPath);
  const processedLines = validLines.slice(0, manifest.lastProcessedLine);
  const hash = computeHash(processedLines);

  if (hash !== manifest.contentHash) {
    return { valid: false, reason: 'hash_mismatch' };
  }

  if (manifest.embeddingModel !== process.env.ONNX_EMBEDDING_MODEL) {
    return { valid: false, reason: 'model_mismatch' };
  }

  if (manifest.modelPrecision !== process.env.ONNX_EMBEDDING_MODEL_PRECISION) {
    return { valid: false, reason: 'precision_mismatch' };
  }

  return { valid: true, reason: null };
}

/**
 * Get new (unprocessed) lines from the JSONL file.
 * Returns parsed JSON objects for lines beyond lastProcessedLine.
 */
export function getNewLines(jsonlPath, lastProcessedLine) {
  const validLines = parseJsonlLines(jsonlPath);
  return validLines.slice(lastProcessedLine);
}

/**
 * Update the manifest after an incremental run.
 */
export function updateManifest(manifestPath, newLineCount, jsonlPath, model, precision) {
  const validLines = parseJsonlLines(jsonlPath);
  const processedLines = validLines.slice(0, newLineCount);
  const manifest = {
    lastProcessedLine: newLineCount,
    contentHash: computeHash(processedLines),
    embeddingModel: model,
    modelPrecision: precision,
    lastRunDate: new Date().toISOString()
  };
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  return manifest;
}
