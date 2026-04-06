// -----------------------------------------------------------
// -- Tests for embedding-utils v0.3.0 features used by FTA --
// -----------------------------------------------------------
import {
  assignToCluster,
  silhouetteScore,
  hdbscan,
  clusterEmbeddings,
  averageEmbeddings,
  batchIncrementalAverage,
  cosineSimilarity,
  centroidCohesion,
} from 'embedding-utils';
import assert from 'assert';

console.log('Running v0.3.0 feature tests...');

// ================================
// == assignToCluster tests      ==
// ================================

function testAssignToClusterBasic() {
  console.log('\nTest 1: assignToCluster - basic assignment');

  // Two well-separated clusters
  const clusters = [
    {
      centroid: [1.0, 0.0],
      members: [[1.0, 0.0], [0.9, 0.1]],
      size: 2,
      cohesion: 1.0,
    },
    {
      centroid: [0.0, 1.0],
      members: [[0.0, 1.0], [0.1, 0.9]],
      size: 2,
      cohesion: 1.0,
    },
  ];

  // Embedding close to cluster 0
  const result0 = assignToCluster([0.95, 0.05], clusters);
  assert.strictEqual(result0.clusterIndex, 0, 'Should assign to cluster 0');
  assert(result0.similarity > 0.9, 'Similarity to cluster 0 should be high');

  // Embedding close to cluster 1
  const result1 = assignToCluster([0.05, 0.95], clusters);
  assert.strictEqual(result1.clusterIndex, 1, 'Should assign to cluster 1');
  assert(result1.similarity > 0.9, 'Similarity to cluster 1 should be high');

  console.log('  assignToCluster basic test passed');
}

function testAssignToClusterReturnShape() {
  console.log('\nTest 2: assignToCluster - return value structure');

  const clusters = [
    { centroid: [1.0, 0.0], members: [[1.0, 0.0]], size: 1, cohesion: 1.0 },
  ];

  const result = assignToCluster([0.8, 0.2], clusters);

  assert(typeof result === 'object', 'Should return an object');
  assert('clusterIndex' in result, 'Should have clusterIndex property');
  assert('similarity' in result, 'Should have similarity property');
  assert(typeof result.clusterIndex === 'number', 'clusterIndex should be number');
  assert(typeof result.similarity === 'number', 'similarity should be number');
  assert(result.clusterIndex >= 0, 'clusterIndex should be non-negative');

  console.log('  assignToCluster return shape test passed');
}

function testAssignToClusterWithThreshold() {
  console.log('\nTest 3: assignToCluster - threshold enforcement');

  const clusters = [
    { centroid: [1.0, 0.0], members: [[1.0, 0.0]], size: 1, cohesion: 1.0 },
    { centroid: [0.0, 1.0], members: [[0.0, 1.0]], size: 1, cohesion: 1.0 },
  ];

  // Embedding roughly between both clusters - low similarity to either
  const result = assignToCluster([0.7071, 0.7071], clusters, { threshold: 0.95 });
  assert.strictEqual(result.clusterIndex, -1, 'Should return -1 when below threshold');
  assert(result.similarity < 0.95, 'Similarity should be below threshold');

  // Same embedding without threshold - should assign to best match
  const resultNoThreshold = assignToCluster([0.7071, 0.7071], clusters);
  assert(resultNoThreshold.clusterIndex >= 0, 'Should assign when no threshold');

  console.log('  assignToCluster threshold test passed');
}

function testAssignToClusterWithAdapters() {
  console.log('\nTest 4: assignToCluster - lightweight adapter objects (FTA incremental pattern)');

  // Simulate FTA's incremental mode: cluster data loaded from JSON
  const clusterData = [
    { filename: 'topic-cluster-1.json', data: { embedding: [1.0, 0.0], clusterSize: 10 } },
    { filename: 'topic-cluster-2.json', data: { embedding: [0.0, 1.0], clusterSize: 5 } },
  ];

  // Build adapter objects like generate.js does
  const clusterAdapters = clusterData.map(c => ({ centroid: c.data.embedding }));

  const { clusterIndex } = assignToCluster([0.9, 0.1], clusterAdapters);
  assert.strictEqual(clusterIndex, 0, 'Should assign to first cluster via adapter');

  const { clusterIndex: idx2 } = assignToCluster([0.1, 0.9], clusterAdapters);
  assert.strictEqual(idx2, 1, 'Should assign to second cluster via adapter');

  console.log('  assignToCluster adapter pattern test passed');
}

// ================================
// == silhouetteScore tests      ==
// ================================

function testSilhouetteScoreWellSeparated() {
  console.log('\nTest 5: silhouetteScore - well-separated clusters');

  const clusters = clusterEmbeddings([
    [1.0, 0.0], [0.95, 0.05], [0.9, 0.1],
    [0.0, 1.0], [0.05, 0.95], [0.1, 0.9],
  ], { similarityThreshold: 0.9, minClusterSize: 2, maxClusters: 3 });

  assert(clusters.length >= 2, 'Should have at least 2 clusters');

  const score = silhouetteScore(clusters);
  assert(typeof score === 'number', 'silhouetteScore should return a number');
  assert(score >= -1 && score <= 1, 'Score should be in [-1, 1]');
  assert(score > 0, 'Well-separated clusters should have positive silhouette');

  console.log(`  silhouetteScore = ${score.toFixed(4)} (well-separated)`);
  console.log('  silhouetteScore well-separated test passed');
}

function testSilhouetteScoreSingleCluster() {
  console.log('\nTest 6: silhouetteScore - single cluster returns 0');

  const clusters = [{
    centroid: [0.5, 0.5],
    members: [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
    size: 3,
    cohesion: 0.7,
  }];

  const score = silhouetteScore(clusters);
  assert.strictEqual(score, 0, 'Single cluster should return 0');

  console.log('  silhouetteScore single cluster test passed');
}

function testSilhouetteScoreInGeneratePipeline() {
  console.log('\nTest 7: silhouetteScore - integration with clustering pipeline');

  const embeddings = [
    [1.0, 0.0], [0.9, 0.1], [0.85, 0.15],
    [0.0, 1.0], [0.1, 0.9], [0.15, 0.85],
  ];

  const clusters = clusterEmbeddings(embeddings, {
    similarityThreshold: 0.9,
    minClusterSize: 2,
    maxClusters: 3,
  });

  // Simulate generate.js pattern
  const silhouette = clusters.length >= 2 ? silhouetteScore(clusters) : 0;

  // Verify it can be formatted like generate.js does
  const formatted = silhouette.toFixed(4);
  assert(typeof formatted === 'string', 'Should format to string');
  assert(formatted.length > 0, 'Formatted string should not be empty');

  console.log(`  silhouetteScore in pipeline = ${formatted}`);
  console.log('  silhouetteScore pipeline integration test passed');
}

// ================================
// == HDBSCAN tests              ==
// ================================

function testHdbscanBasic() {
  console.log('\nTest 8: HDBSCAN - basic clustering');

  // Create embeddings with 2 dense clusters (6D for better density estimation)
  const cluster1 = Array.from({ length: 10 }, (_, i) => {
    const base = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    base[1] = (i * 0.01);
    return base;
  });
  const cluster2 = Array.from({ length: 10 }, (_, i) => {
    const base = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    base[4] = (i * 0.01);
    return base;
  });
  const embeddings = [...cluster1, ...cluster2];

  const result = hdbscan(embeddings, { minClusterSize: 3 });

  // Verify result shape
  assert(Array.isArray(result.clusters), 'result.clusters should be an array');
  assert('noise' in result, 'result should have noise property');
  assert(Array.isArray(result.noise.members), 'noise.members should be an array');
  assert(Array.isArray(result.noise.indices), 'noise.indices should be an array');
  assert(Array.isArray(result.labels), 'result.labels should be an array');
  assert.strictEqual(result.labels.length, embeddings.length, 'labels should match input length');

  // Each cluster should have centroid, members, size
  for (const cluster of result.clusters) {
    assert('centroid' in cluster, 'Cluster should have centroid');
    assert('members' in cluster, 'Cluster should have members');
    assert('size' in cluster, 'Cluster should have size');
    assert('indices' in cluster, 'HDBSCAN cluster should have indices');
    assert.strictEqual(cluster.members.length, cluster.size, 'members.length should equal size');
  }

  console.log(`  HDBSCAN found ${result.clusters.length} clusters, ${result.noise.members.length} noise points`);
  console.log('  HDBSCAN basic test passed');
}

function testHdbscanWithLabels() {
  console.log('\nTest 9: HDBSCAN - label preservation');

  const embeddings = Array.from({ length: 12 }, (_, i) => {
    if (i < 6) return [1.0, i * 0.01, 0.0, 0.0];
    return [0.0, 0.0, 1.0, (i - 6) * 0.01];
  });
  const labels = embeddings.map((_, i) => `phrase-${i}`);

  const result = hdbscan(embeddings, { minClusterSize: 3, labels });

  // Verify labels are present in clusters and noise
  for (const cluster of result.clusters) {
    if (cluster.labels) {
      assert.strictEqual(cluster.labels.length, cluster.members.length,
        'Cluster labels should match members count');
    }
  }
  if (result.noise.labels) {
    assert.strictEqual(result.noise.labels.length, result.noise.members.length,
      'Noise labels should match noise members count');
  }

  // All input labels should be accounted for
  const allLabels = [];
  for (const cluster of result.clusters) {
    if (cluster.labels) allLabels.push(...cluster.labels);
  }
  if (result.noise.labels) allLabels.push(...result.noise.labels);
  assert.strictEqual(allLabels.length, labels.length,
    'All labels should be preserved across clusters + noise');

  console.log('  HDBSCAN label preservation test passed');
}

function testHdbscanNoiseReassignment() {
  console.log('\nTest 10: HDBSCAN - noise point reassignment (FTA pattern)');

  // Use high minClusterSize to force noise points
  const embeddings = [
    [1.0, 0.0], [0.95, 0.05], [0.9, 0.1], [0.85, 0.15], [0.8, 0.2],
    [0.0, 1.0], [0.05, 0.95], [0.1, 0.9], [0.15, 0.85], [0.2, 0.8],
    [0.5, 0.5], // outlier
  ];
  const labels = embeddings.map((_, i) => `phrase-${i}`);

  const result = hdbscan(embeddings, { minClusterSize: 4, labels });

  // If there are clusters and noise, simulate FTA's reassignment pattern
  if (result.clusters.length > 0 && result.noise.members.length > 0) {
    const clusters = result.clusters;
    const originalTotalSize = clusters.reduce((s, c) => s + c.size, 0);
    const noiseCount = result.noise.members.length;

    // Reassign noise to nearest cluster (replicating generate.js lines 237-251)
    for (let n = 0; n < result.noise.members.length; n++) {
      const { clusterIndex } = assignToCluster(result.noise.members[n], clusters);
      assert(clusterIndex >= 0, 'Noise point should be assignable to a cluster');
      clusters[clusterIndex].members.push(result.noise.members[n]);
      if (result.noise.labels && result.noise.labels[n]) {
        clusters[clusterIndex].labels = clusters[clusterIndex].labels || [];
        clusters[clusterIndex].labels.push(result.noise.labels[n]);
      }
      clusters[clusterIndex].size += 1;
    }

    // Recompute centroids
    for (const cluster of clusters) {
      cluster.centroid = averageEmbeddings(cluster.members);
      cluster.cohesion = centroidCohesion(cluster);
    }

    // Verify all points accounted for
    const newTotalSize = clusters.reduce((s, c) => s + c.size, 0);
    assert.strictEqual(newTotalSize, originalTotalSize + noiseCount,
      'All noise points should be absorbed into clusters');
    assert.strictEqual(newTotalSize, embeddings.length,
      'Total assigned should equal total input');

    console.log(`  Reassigned ${noiseCount} noise points into ${clusters.length} clusters`);
  } else {
    console.log('  (No noise points to reassign in this run)');
  }

  console.log('  HDBSCAN noise reassignment test passed');
}

function testHdbscanFallbackNoCluster() {
  console.log('\nTest 11: HDBSCAN - fallback when no clusters found');

  // Very high minClusterSize forces all points to noise
  const embeddings = [
    [1.0, 0.0],
    [0.0, 1.0],
    [0.5, 0.5],
  ];

  const result = hdbscan(embeddings, { minClusterSize: 100 });

  // Simulate FTA's fallback (generate.js lines 254-263)
  let clusters;
  if (result.clusters.length === 0) {
    clusters = [{
      centroid: averageEmbeddings(embeddings),
      members: embeddings.map(e => e),
      size: embeddings.length,
      cohesion: 1.0,
    }];
  } else {
    clusters = result.clusters;
  }

  assert.strictEqual(clusters.length, 1, 'Fallback should produce single cluster');
  assert.strictEqual(clusters[0].size, embeddings.length, 'Single cluster should contain all embeddings');
  assert(clusters[0].centroid instanceof Float32Array, 'Centroid should be Float32Array from averageEmbeddings');

  console.log('  HDBSCAN fallback test passed');
}

// ================================
// == Float32Array handling tests ==
// ================================

function testFloat32ArrayFromAverageEmbeddings() {
  console.log('\nTest 12: Float32Array - averageEmbeddings returns Float32Array');

  const vectors = [[1, 2, 3], [4, 5, 6]];
  const result = averageEmbeddings(vectors);

  assert(result instanceof Float32Array, 'averageEmbeddings should return Float32Array');
  assert.deepStrictEqual(Array.from(result), [2.5, 3.5, 4.5], 'Values should be correct');

  // Verify JSON serialization requires Array.from()
  const json = JSON.stringify(Array.from(result));
  assert(json.startsWith('['), 'Array.from() + JSON.stringify should produce array');
  const parsed = JSON.parse(json);
  assert(Array.isArray(parsed), 'Parsed result should be array');

  console.log('  Float32Array averageEmbeddings test passed');
}

function testFloat32ArrayFromBatchIncrementalAverage() {
  console.log('\nTest 13: Float32Array - batchIncrementalAverage returns Float32Array');

  const existing = [2.0, 3.0, 4.0];
  const newBatch = [[5.0, 6.0, 7.0]];
  const result = batchIncrementalAverage(existing, newBatch, 1);

  assert(result instanceof Float32Array, 'batchIncrementalAverage should return Float32Array');

  // Verify Array.from() conversion for serialization
  const converted = Array.from(result);
  assert(Array.isArray(converted), 'Array.from() should produce plain array');

  const json = JSON.stringify(converted);
  assert(json.startsWith('['), 'Should serialize as JSON array');

  console.log('  Float32Array batchIncrementalAverage test passed');
}

function testFloat32ArrayClusterCentroids() {
  console.log('\nTest 14: Float32Array - clusterEmbeddings returns Float32Array centroids');

  const embeddings = [
    [1.0, 0.0], [0.9, 0.1],
    [0.0, 1.0], [0.1, 0.9],
  ];

  const clusters = clusterEmbeddings(embeddings, {
    similarityThreshold: 0.9,
    minClusterSize: 1,
    maxClusters: 2,
  });

  for (const cluster of clusters) {
    assert(cluster.centroid instanceof Float32Array,
      'Cluster centroid should be Float32Array');

    // Verify safe serialization pattern
    const safe = Array.from(cluster.centroid);
    const json = JSON.stringify(safe);
    assert(json.startsWith('['), 'Converted centroid should serialize as JSON array');
  }

  console.log('  Float32Array cluster centroids test passed');
}

function testFloat32ArrayIncrementalPipeline() {
  console.log('\nTest 15: Float32Array - full incremental pipeline conversion chain');

  // Simulate the complete incremental flow from generate.js
  // 1. Load cluster data from "JSON" (plain arrays)
  const clusterData = [
    { embedding: [0.9, 0.1], clusterSize: 5 },
    { embedding: [0.1, 0.9], clusterSize: 3 },
  ];

  // 2. Build adapters for assignToCluster
  const adapters = clusterData.map(c => ({ centroid: c.embedding }));

  // 3. New embedding arrives
  const newEmbedding = [0.85, 0.15];

  // 4. Assign to cluster
  const { clusterIndex } = assignToCluster(newEmbedding, adapters);
  assert.strictEqual(clusterIndex, 0, 'Should assign to first cluster');

  // 5. Update centroid with batchIncrementalAverage (returns Float32Array)
  const updatedCentroid = batchIncrementalAverage(
    clusterData[clusterIndex].embedding,
    [newEmbedding],
    clusterData[clusterIndex].clusterSize
  );
  assert(updatedCentroid instanceof Float32Array, 'Updated centroid should be Float32Array');

  // 6. Convert for JSON storage (critical step)
  const forJson = Array.from(updatedCentroid);
  assert(Array.isArray(forJson), 'Should convert to plain array');

  // 7. Simulate JSON round-trip
  const serialized = JSON.stringify({ embedding: forJson });
  const deserialized = JSON.parse(serialized);
  assert(Array.isArray(deserialized.embedding), 'After JSON round-trip, embedding should be array');
  assert.strictEqual(deserialized.embedding.length, 2, 'Should preserve dimensions');

  // 8. Update adapter for next iteration
  adapters[clusterIndex].centroid = forJson;
  const { clusterIndex: nextIdx } = assignToCluster([0.88, 0.12], adapters);
  assert.strictEqual(nextIdx, 0, 'Updated adapter should still work for assignment');

  console.log('  Float32Array incremental pipeline test passed');
}

function testFloat32ArrayCosineSimilarityAcceptsBoth() {
  console.log('\nTest 16: Float32Array - cosineSimilarity accepts both types');

  const plainArray = [1.0, 0.0];
  const float32 = new Float32Array([0.9, 0.1]);

  // All combinations should work
  const sim1 = cosineSimilarity(plainArray, plainArray);
  const sim2 = cosineSimilarity(float32, float32);
  const sim3 = cosineSimilarity(plainArray, float32);
  const sim4 = cosineSimilarity(float32, plainArray);

  assert(typeof sim1 === 'number' && !isNaN(sim1), 'array vs array should work');
  assert(typeof sim2 === 'number' && !isNaN(sim2), 'Float32 vs Float32 should work');
  assert(typeof sim3 === 'number' && !isNaN(sim3), 'array vs Float32 should work');
  assert(typeof sim4 === 'number' && !isNaN(sim4), 'Float32 vs array should work');

  // Results should be consistent
  const TOLERANCE = 1e-6;
  assert(Math.abs(sim3 - sim4) < TOLERANCE, 'Mixed-type results should match');

  console.log('  Float32Array cosineSimilarity compatibility test passed');
}

// ================================
// == Run all tests              ==
// ================================
try {
  // assignToCluster
  testAssignToClusterBasic();
  testAssignToClusterReturnShape();
  testAssignToClusterWithThreshold();
  testAssignToClusterWithAdapters();

  // silhouetteScore
  testSilhouetteScoreWellSeparated();
  testSilhouetteScoreSingleCluster();
  testSilhouetteScoreInGeneratePipeline();

  // HDBSCAN
  testHdbscanBasic();
  testHdbscanWithLabels();
  testHdbscanNoiseReassignment();
  testHdbscanFallbackNoCluster();

  // Float32Array
  testFloat32ArrayFromAverageEmbeddings();
  testFloat32ArrayFromBatchIncrementalAverage();
  testFloat32ArrayClusterCentroids();
  testFloat32ArrayIncrementalPipeline();
  testFloat32ArrayCosineSimilarityAcceptsBoth();

  console.log('\n\u2705 All v0.3.0 feature tests passed!');
} catch (error) {
  console.error('\n\u274C Test failed:', error);
  process.exit(1);
}
