// ---------------------------------------------------
// -- Incremental Integration Test                  --
// -- Validates FR-5 and NFR-1: math equivalence    --
// ---------------------------------------------------
import assert from 'assert';
import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';

console.log('Running incremental integration test...');
console.log('(This test runs full generation twice + incremental -- may take a few minutes)\n');

const DATA_DIR = 'data';
const JSONL_PATH = path.join(DATA_DIR, 'training_data.jsonl');
const MANIFEST_PATH = path.join(DATA_DIR, 'incremental-manifest.json');
const EMBEDDINGS_DIR = path.join(DATA_DIR, 'topic_embeddings');

// -- Test data: 6 phrases across 2 topics (disney, llamas) --
// Using topics from labels-config.js so generate.js processes them
const allPhrases = [
  { label: 'disney', text: 'Mickey Mouse is a classic Disney character' },
  { label: 'disney', text: 'Walt Disney created magical theme parks' },
  { label: 'disney', text: 'The Lion King is an animated Disney film' },
  { label: 'llamas', text: 'Llamas are domesticated South American animals' },
  { label: 'llamas', text: 'Alpacas and llamas are related camelids' },
  { label: 'llamas', text: 'Llamas are used as pack animals in the Andes' },
];

const firstBatch = allPhrases.slice(0, 4); // 2 disney + 2 llamas
const appendBatch = allPhrases.slice(4);   // 1 disney + 1 llamas

// -- Backup existing data --
let backupDir = null;

function backupData() {
  backupDir = fs.mkdtempSync(path.join('data', 'integration-test-backup-'));
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
  // Restore JSONL
  const backupJsonl = path.join(backupDir, 'training_data.jsonl');
  if (fs.existsSync(backupJsonl)) {
    fs.copyFileSync(backupJsonl, JSONL_PATH);
  }

  // Restore manifest
  const backupManifest = path.join(backupDir, 'manifest.json');
  if (fs.existsSync(backupManifest)) {
    fs.copyFileSync(backupManifest, MANIFEST_PATH);
  } else if (fs.existsSync(MANIFEST_PATH)) {
    fs.unlinkSync(MANIFEST_PATH);
  }

  // Restore embeddings
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

  // Cleanup backup
  fs.rmSync(backupDir, { recursive: true });
}

function writeJsonl(phrases) {
  const content = phrases.map(p => JSON.stringify(p)).join('\n');
  fs.writeFileSync(JSONL_PATH, content);
}

function loadCentroids() {
  const centroids = {};
  const files = fs.readdirSync(EMBEDDINGS_DIR).filter(f => f.endsWith('.json'));
  for (const file of files) {
    const data = JSON.parse(fs.readFileSync(path.join(EMBEDDINGS_DIR, file), 'utf8'));
    centroids[file] = data;
  }
  return centroids;
}

function runGenerate(flags = '') {
  execSync(`node generate.js --preset legacy ${flags}`, {
    stdio: 'pipe',
    timeout: 300000
  });
}

try {
  backupData();

  // Step 1: Full gen on ALL 6 phrases → reference centroids
  console.log('Step 1: Full generation on all 6 phrases (reference)...');
  writeJsonl(allPhrases);
  runGenerate();
  const referenceCentroids = loadCentroids();
  console.log(`  Reference: ${Object.keys(referenceCentroids).length} cluster files`);

  // Step 2: Full gen on first 4 phrases (creates manifest)
  console.log('Step 2: Full generation on first 4 phrases...');
  writeJsonl(firstBatch);
  runGenerate();
  console.log('  Manifest created');

  // Step 3: Append remaining 2 phrases
  console.log('Step 3: Appending 2 new phrases...');
  const appendContent = '\n' + appendBatch.map(p => JSON.stringify(p)).join('\n');
  fs.appendFileSync(JSONL_PATH, appendContent);

  // Step 4: Run incremental
  console.log('Step 4: Running incremental generation...');
  runGenerate('--incremental');

  // Step 5: Compare centroids
  console.log('Step 5: Comparing centroids...');
  const incrementalCentroids = loadCentroids();

  const refKeys = Object.keys(referenceCentroids).sort();
  const incKeys = Object.keys(incrementalCentroids).sort();

  assert.deepStrictEqual(incKeys, refKeys, 'Should have same cluster files');

  const TOLERANCE = 1e-10;
  let allMatch = true;

  for (const file of refKeys) {
    const refEmb = referenceCentroids[file].embedding;
    const incEmb = incrementalCentroids[file].embedding;

    assert.strictEqual(refEmb.length, incEmb.length, `Embedding dimensions should match for ${file}`);

    for (let i = 0; i < refEmb.length; i++) {
      const diff = Math.abs(refEmb[i] - incEmb[i]);
      if (diff >= TOLERANCE) {
        console.log(`  Mismatch in ${file}, dimension ${i}: ref=${refEmb[i]}, inc=${incEmb[i]}, diff=${diff}`);
        allMatch = false;
      }
    }
  }

  assert.ok(allMatch, 'All centroid dimensions should match within tolerance');

  console.log('\n✅ Incremental integration test passed! Centroids match reference within tolerance.');
} catch (error) {
  console.error('\n❌ Integration test failed:', error.message || error);
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
