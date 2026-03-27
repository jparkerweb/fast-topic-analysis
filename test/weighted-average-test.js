// -------------------------------------------
// -- Weighted Average Math Unit Tests      --
// -------------------------------------------
import { weightedAverage } from '../modules/embedding.js';
import assert from 'assert';

console.log('Running weighted average tests...');

// Test 1: 5 embeddings - full average vs incremental weighted average
function testIncrementalMatchesFullAverage() {
  console.log('\nTest 1: Incremental weighted average matches full average');

  const e1 = [0.1, 0.2, 0.3, 0.4];
  const e2 = [0.5, 0.6, 0.7, 0.8];
  const e3 = [0.9, 1.0, 1.1, 1.2];
  const e4 = [1.3, 1.4, 1.5, 1.6];
  const e5 = [1.7, 1.8, 1.9, 2.0];

  // Method 1: Manual full average of all 5
  const all = [e1, e2, e3, e4, e5];
  const fullAvg = all[0].map((_, i) => {
    const sum = all.reduce((acc, emb) => acc + emb[i], 0);
    return sum / all.length;
  });

  // Method 2: Average first 3, then weightedAverage with remaining 2
  const first3 = [e1, e2, e3];
  const avg3 = first3[0].map((_, i) => {
    const sum = first3.reduce((acc, emb) => acc + emb[i], 0);
    return sum / first3.length;
  });
  const incrementalResult = weightedAverage(avg3, 3, [e4, e5]);

  // Assert each dimension matches within tolerance
  for (let i = 0; i < fullAvg.length; i++) {
    assert.ok(
      Math.abs(incrementalResult[i] - fullAvg[i]) < 1e-10,
      `Dimension ${i}: expected ${fullAvg[i]}, got ${incrementalResult[i]}`
    );
  }

  console.log('✓ Incremental weighted average matches full average');
}

// Test 2: Single embedding - weightedAverage(e1, 1, [e2]) equals average of [e1, e2]
function testSingleEmbedding() {
  console.log('\nTest 2: Single embedding weighted average');

  const e1 = [0.3, 0.7, 0.1, 0.9];
  const e2 = [0.8, 0.2, 0.6, 0.4];

  // Manual average of [e1, e2]
  const expected = e1.map((v, i) => (v + e2[i]) / 2);

  const result = weightedAverage(e1, 1, [e2]);

  for (let i = 0; i < expected.length; i++) {
    assert.ok(
      Math.abs(result[i] - expected[i]) < 1e-10,
      `Dimension ${i}: expected ${expected[i]}, got ${result[i]}`
    );
  }

  console.log('✓ Single embedding weighted average correct');
}

// Test 3: Large count - weightedAverage(centroid, 100, [newEmbed])
function testLargeCount() {
  console.log('\nTest 3: Large count weighted average');

  const centroid = [0.5, 0.5, 0.5, 0.5];
  const newEmbed = [1.0, 0.0, 1.0, 0.0];

  const result = weightedAverage(centroid, 100, [newEmbed]);

  // Expected: (centroid[i] * 100 + newEmbed[i]) / 101
  for (let i = 0; i < centroid.length; i++) {
    const expected = (centroid[i] * 100 + newEmbed[i]) / 101;
    assert.ok(
      Math.abs(result[i] - expected) < 1e-10,
      `Dimension ${i}: expected ${expected}, got ${result[i]}`
    );
  }

  console.log('✓ Large count weighted average correct');
}

// Run all tests
try {
  testIncrementalMatchesFullAverage();
  testSingleEmbedding();
  testLargeCount();

  console.log('\n✅ All weighted average tests passed!');
} catch (error) {
  console.error('\n❌ Test failed:', error);
  process.exit(1);
}
