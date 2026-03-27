// -------------------------------------------
// -- Manifest Validation Unit Tests        --
// -------------------------------------------
import { createManifest, loadManifest, validateManifest, getNewLines } from '../modules/manifest.js';
import assert from 'assert';
import fs from 'fs';
import os from 'os';
import path from 'path';

console.log('Running manifest tests...');

// Helper: create temp directory
function createTempDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'manifest-test-'));
}

// Helper: create a JSONL file with N lines
function createJsonlFile(dir, lines) {
  const filePath = path.join(dir, 'test.jsonl');
  const content = lines.map(l => JSON.stringify(l)).join('\n');
  fs.writeFileSync(filePath, content);
  return filePath;
}

// Save original env vars
const origModel = process.env.ONNX_EMBEDDING_MODEL;
const origPrecision = process.env.ONNX_EMBEDDING_MODEL_PRECISION;

// ========================
// Manifest Validation Tests
// ========================

// Test 1: Valid manifest passes validation
function testValidManifest() {
  console.log('\nTest 1: Valid manifest passes validation');

  const tmpDir = createTempDir();
  const lines = [
    { label: 'test', text: 'phrase 1' },
    { label: 'test', text: 'phrase 2' },
    { label: 'test', text: 'phrase 3' },
    { label: 'test', text: 'phrase 4' },
    { label: 'test', text: 'phrase 5' },
  ];
  const jsonlPath = createJsonlFile(tmpDir, lines);
  const manifestPath = path.join(tmpDir, 'manifest.json');

  process.env.ONNX_EMBEDDING_MODEL = 'test-model';
  process.env.ONNX_EMBEDDING_MODEL_PRECISION = 'fp32';

  createManifest(jsonlPath, manifestPath, 'test-model', 'fp32');
  const manifest = loadManifest(manifestPath);
  const result = validateManifest(manifest, jsonlPath);

  assert.deepStrictEqual(result, { valid: true, reason: null }, 'Should be valid');

  fs.rmSync(tmpDir, { recursive: true });
  console.log('✓ Valid manifest passes validation');
}

// Test 2: Hash mismatch after modifying JSONL
function testHashMismatch() {
  console.log('\nTest 2: Hash mismatch after modifying JSONL');

  const tmpDir = createTempDir();
  const lines = [
    { label: 'test', text: 'phrase 1' },
    { label: 'test', text: 'phrase 2' },
    { label: 'test', text: 'phrase 3' },
    { label: 'test', text: 'phrase 4' },
    { label: 'test', text: 'phrase 5' },
  ];
  const jsonlPath = createJsonlFile(tmpDir, lines);
  const manifestPath = path.join(tmpDir, 'manifest.json');

  process.env.ONNX_EMBEDDING_MODEL = 'test-model';
  process.env.ONNX_EMBEDDING_MODEL_PRECISION = 'fp32';

  createManifest(jsonlPath, manifestPath, 'test-model', 'fp32');
  const manifest = loadManifest(manifestPath);

  // Modify line 3 of the JSONL
  lines[2] = { label: 'test', text: 'MODIFIED phrase 3' };
  createJsonlFile(tmpDir, lines);

  const result = validateManifest(manifest, jsonlPath);
  assert.deepStrictEqual(result, { valid: false, reason: 'hash_mismatch' }, 'Should detect hash mismatch');

  fs.rmSync(tmpDir, { recursive: true });
  console.log('✓ Hash mismatch detected');
}

// Test 3: Model mismatch
function testModelMismatch() {
  console.log('\nTest 3: Model mismatch');

  const tmpDir = createTempDir();
  const lines = [
    { label: 'test', text: 'phrase 1' },
    { label: 'test', text: 'phrase 2' },
    { label: 'test', text: 'phrase 3' },
    { label: 'test', text: 'phrase 4' },
    { label: 'test', text: 'phrase 5' },
  ];
  const jsonlPath = createJsonlFile(tmpDir, lines);
  const manifestPath = path.join(tmpDir, 'manifest.json');

  process.env.ONNX_EMBEDDING_MODEL = 'model-a';
  process.env.ONNX_EMBEDDING_MODEL_PRECISION = 'fp32';

  createManifest(jsonlPath, manifestPath, 'model-a', 'fp32');
  const manifest = loadManifest(manifestPath);

  // Change env to a different model
  process.env.ONNX_EMBEDDING_MODEL = 'model-b';

  const result = validateManifest(manifest, jsonlPath);
  assert.deepStrictEqual(result, { valid: false, reason: 'model_mismatch' }, 'Should detect model mismatch');

  fs.rmSync(tmpDir, { recursive: true });
  console.log('✓ Model mismatch detected');
}

// Test 4: Precision mismatch
function testPrecisionMismatch() {
  console.log('\nTest 4: Precision mismatch');

  const tmpDir = createTempDir();
  const lines = [
    { label: 'test', text: 'phrase 1' },
    { label: 'test', text: 'phrase 2' },
    { label: 'test', text: 'phrase 3' },
    { label: 'test', text: 'phrase 4' },
    { label: 'test', text: 'phrase 5' },
  ];
  const jsonlPath = createJsonlFile(tmpDir, lines);
  const manifestPath = path.join(tmpDir, 'manifest.json');

  process.env.ONNX_EMBEDDING_MODEL = 'test-model';
  process.env.ONNX_EMBEDDING_MODEL_PRECISION = 'fp32';

  createManifest(jsonlPath, manifestPath, 'test-model', 'fp32');
  const manifest = loadManifest(manifestPath);

  // Change env to a different precision
  process.env.ONNX_EMBEDDING_MODEL_PRECISION = 'fp16';

  const result = validateManifest(manifest, jsonlPath);
  assert.deepStrictEqual(result, { valid: false, reason: 'precision_mismatch' }, 'Should detect precision mismatch');

  fs.rmSync(tmpDir, { recursive: true });
  console.log('✓ Precision mismatch detected');
}

// ========================
// getNewLines Tests
// ========================

// Test 5: getNewLines returns correct new lines
function testGetNewLinesPartial() {
  console.log('\nTest 5: getNewLines returns lines after lastProcessedLine');

  const tmpDir = createTempDir();
  const lines = [
    { label: 'test', text: 'phrase 1' },
    { label: 'test', text: 'phrase 2' },
    { label: 'test', text: 'phrase 3' },
    { label: 'test', text: 'phrase 4' },
    { label: 'test', text: 'phrase 5' },
  ];
  // Add a blank line to test filtering
  const jsonlPath = path.join(tmpDir, 'test.jsonl');
  const content = lines.map(l => JSON.stringify(l)).join('\n') + '\n\n';
  fs.writeFileSync(jsonlPath, content);

  const result = getNewLines(jsonlPath, 3);
  assert.strictEqual(result.length, 2, 'Should return 2 new lines');
  assert.strictEqual(result[0].text, 'phrase 4');
  assert.strictEqual(result[1].text, 'phrase 5');
  assert.strictEqual(result[0].label, 'test');
  assert.strictEqual(result[1].label, 'test');

  fs.rmSync(tmpDir, { recursive: true });
  console.log('✓ getNewLines returns correct partial lines');
}

// Test 6: getNewLines returns empty when all processed
function testGetNewLinesEmpty() {
  console.log('\nTest 6: getNewLines returns empty when all processed');

  const tmpDir = createTempDir();
  const lines = [
    { label: 'test', text: 'phrase 1' },
    { label: 'test', text: 'phrase 2' },
    { label: 'test', text: 'phrase 3' },
    { label: 'test', text: 'phrase 4' },
    { label: 'test', text: 'phrase 5' },
  ];
  const jsonlPath = createJsonlFile(tmpDir, lines);

  const result = getNewLines(jsonlPath, 5);
  assert.strictEqual(result.length, 0, 'Should return empty array');

  fs.rmSync(tmpDir, { recursive: true });
  console.log('✓ getNewLines returns empty when all processed');
}

// Test 7: getNewLines returns all when lastProcessedLine is 0
function testGetNewLinesAll() {
  console.log('\nTest 7: getNewLines returns all lines when lastProcessedLine is 0');

  const tmpDir = createTempDir();
  const lines = [
    { label: 'test', text: 'phrase 1' },
    { label: 'test', text: 'phrase 2' },
    { label: 'test', text: 'phrase 3' },
    { label: 'test', text: 'phrase 4' },
    { label: 'test', text: 'phrase 5' },
  ];
  const jsonlPath = createJsonlFile(tmpDir, lines);

  const result = getNewLines(jsonlPath, 0);
  assert.strictEqual(result.length, 5, 'Should return all 5 lines');
  assert.strictEqual(result[0].text, 'phrase 1');
  assert.strictEqual(result[4].text, 'phrase 5');

  fs.rmSync(tmpDir, { recursive: true });
  console.log('✓ getNewLines returns all lines from start');
}

// Run all tests
try {
  testValidManifest();
  testHashMismatch();
  testModelMismatch();
  testPrecisionMismatch();
  testGetNewLinesPartial();
  testGetNewLinesEmpty();
  testGetNewLinesAll();

  console.log('\n✅ All manifest tests passed!');
} catch (error) {
  console.error('\n❌ Test failed:', error);
  process.exit(1);
} finally {
  // Restore original env vars
  process.env.ONNX_EMBEDDING_MODEL = origModel;
  process.env.ONNX_EMBEDDING_MODEL_PRECISION = origPrecision;
}
