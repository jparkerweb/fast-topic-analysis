// ---------------------------------------------------
// -- Incremental Edge Case Tests                   --
// ---------------------------------------------------
import assert from 'assert';
import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';

console.log('Running incremental edge case tests...');

const DATA_DIR = 'data';
const MANIFEST_PATH = path.join(DATA_DIR, 'incremental-manifest.json');
const JSONL_PATH = path.join(DATA_DIR, 'training_data.jsonl');
const EMBEDDINGS_DIR = path.join(DATA_DIR, 'topic_embeddings');

// -- Backup/restore helpers --
let backupDir = null;

function backupData() {
  backupDir = fs.mkdtempSync(path.join(DATA_DIR, 'edge-test-backup-'));
  if (fs.existsSync(JSONL_PATH)) {
    fs.copyFileSync(JSONL_PATH, path.join(backupDir, 'training_data.jsonl'));
  }
  if (fs.existsSync(MANIFEST_PATH)) {
    fs.copyFileSync(MANIFEST_PATH, path.join(backupDir, 'manifest.json'));
  }
  if (fs.existsSync(EMBEDDINGS_DIR)) {
    const files = fs.readdirSync(EMBEDDINGS_DIR).filter(f => f.endsWith('.json'));
    for (const file of files) {
      fs.copyFileSync(path.join(EMBEDDINGS_DIR, file), path.join(backupDir, file));
    }
  }
}

function restoreData() {
  const backupJsonl = path.join(backupDir, 'training_data.jsonl');
  if (fs.existsSync(backupJsonl)) {
    fs.copyFileSync(backupJsonl, JSONL_PATH);
  }

  const backupManifest = path.join(backupDir, 'manifest.json');
  if (fs.existsSync(backupManifest)) {
    fs.copyFileSync(backupManifest, MANIFEST_PATH);
  } else if (fs.existsSync(MANIFEST_PATH)) {
    fs.unlinkSync(MANIFEST_PATH);
  }

  if (fs.existsSync(EMBEDDINGS_DIR)) {
    const currentFiles = fs.readdirSync(EMBEDDINGS_DIR).filter(f => f.endsWith('.json'));
    for (const file of currentFiles) {
      fs.unlinkSync(path.join(EMBEDDINGS_DIR, file));
    }
  }
  const backupFiles = fs.readdirSync(backupDir).filter(f => f.endsWith('.json') && f !== 'manifest.json');
  for (const file of backupFiles) {
    fs.copyFileSync(path.join(backupDir, file), path.join(EMBEDDINGS_DIR, file));
  }

  fs.rmSync(backupDir, { recursive: true });
}

// Test 1: --incremental with no manifest
function testNoManifest() {
  console.log('\nTest 1: --incremental with no manifest');

  // Ensure no manifest exists
  if (fs.existsSync(MANIFEST_PATH)) {
    fs.unlinkSync(MANIFEST_PATH);
  }

  try {
    execSync('node generate.js --incremental', { stdio: 'pipe', timeout: 30000 });
    assert.fail('Should have exited with non-zero code');
  } catch (error) {
    const output = (error.stdout || '').toString() + (error.stderr || '').toString();
    assert.ok(error.status !== 0, 'Should exit with non-zero code');
    assert.ok(
      output.includes('No manifest found') || output.includes('Run full generation'),
      `Output should mention missing manifest, got: ${output.slice(0, 200)}`
    );
  }

  console.log('✓ No manifest triggers error and clear message');
}

// Test 2: --incremental with model mismatch
function testModelMismatch() {
  console.log('\nTest 2: --incremental with model mismatch in manifest');

  // Create a manifest with wrong model
  const fakeManifest = {
    lastProcessedLine: 1,
    contentHash: 'sha256:fake',
    embeddingModel: 'wrong-model-that-does-not-exist',
    modelPrecision: process.env.ONNX_EMBEDDING_MODEL_PRECISION || 'fp32',
    lastRunDate: new Date().toISOString()
  };
  fs.writeFileSync(MANIFEST_PATH, JSON.stringify(fakeManifest, null, 2));

  // Ensure JSONL exists with at least 1 line
  fs.writeFileSync(JSONL_PATH, JSON.stringify({ label: 'test', text: 'phrase 1' }));

  try {
    execSync('node generate.js --incremental', { stdio: 'pipe', timeout: 30000 });
    assert.fail('Should have exited with non-zero code');
  } catch (error) {
    const output = (error.stdout || '').toString() + (error.stderr || '').toString();
    assert.ok(error.status !== 0, 'Should exit with non-zero code');
    assert.ok(
      output.includes('model changed') || output.includes('model') || output.includes('Run full generation'),
      `Output should mention model issue, got: ${output.slice(0, 200)}`
    );
  }

  console.log('✓ Model mismatch triggers error and clear message');
}

// Test 3: --incremental with no new lines
function testNoNewLines() {
  console.log('\nTest 3: --incremental with no new lines');
  console.log('  (Requires full generation first -- may take a minute)');

  // Write test JSONL
  const phrases = [
    { label: 'disney', text: 'Mickey Mouse is a beloved Disney character' },
    { label: 'llamas', text: 'Llamas are friendly animals from South America' },
  ];
  const content = phrases.map(p => JSON.stringify(p)).join('\n');
  fs.writeFileSync(JSONL_PATH, content);

  // Run full generation to create manifest matching all lines
  execSync('node generate.js --preset legacy', { stdio: 'pipe', timeout: 300000 });

  // Run incremental immediately -- no new lines appended
  try {
    const output = execSync('node generate.js --incremental', {
      stdio: 'pipe',
      timeout: 30000
    }).toString();

    assert.ok(
      output.includes('No new training data') || output.includes('Nothing to update'),
      `Output should indicate nothing to update, got: ${output.slice(0, 200)}`
    );
  } catch (error) {
    // Should NOT fail -- zero exit code expected
    assert.fail(`Should exit with zero code, but got error: ${error.message}`);
  }

  console.log('✓ No new lines exits cleanly with informative message');
}

// Run all tests
try {
  backupData();
  testNoManifest();
  testModelMismatch();
  testNoNewLines();

  console.log('\n✅ All edge case tests passed!');
} catch (error) {
  console.error('\n❌ Edge case test failed:', error.message || error);
  process.exit(1);
} finally {
  try {
    restoreData();
  } catch (cleanupError) {
    console.error('Warning: cleanup failed:', cleanupError.message);
    if (backupDir && fs.existsSync(backupDir)) {
      fs.rmSync(backupDir, { recursive: true });
    }
  }
}
