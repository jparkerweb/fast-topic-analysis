// -------------
// -- imports --
// -------------
import { clusterEmbeddings, calculateAverageEmbedding, updateClusteringConfig } from '../modules/clusterEmbeddings.js';
import { cosineSimilarity } from '../modules/similarity.js';
import assert from 'assert';

// Function to calculate cohesion for testing
function calculateCohesion(embeddings, centroid) {
  let totalSimilarity = 0;
  for (const embedding of embeddings) {
    totalSimilarity += cosineSimilarity(embedding, centroid);
  }
  return embeddings.length > 0 ? totalSimilarity / embeddings.length : 1.0;
}

// -----------------------------
// -- Test clustering function --
// -----------------------------
console.log('Running clustering tests...');

// Test 1: Basic clustering functionality
function testBasicClustering() {
  console.log('\nTest 1: Basic clustering functionality');
  
  // Create some test embeddings (2D for simplicity)
  const embeddings = [
    [1.0, 0.0],  // Cluster 1
    [0.9, 0.1],  // Cluster 1
    [0.8, 0.2],  // Cluster 1
    [0.0, 1.0],  // Cluster 2
    [0.1, 0.9],  // Cluster 2
    [0.2, 0.8],  // Cluster 2
  ];
  
  // Configure clustering for testing
  updateClusteringConfig({
    enableClustering: true,
    similarityThreshold: 0.9,
    minClusterSize: 2,
    maxClusters: 3
  });
  
  // Run clustering
  const clusters = clusterEmbeddings(embeddings);
  
  // Verify we have 2 clusters
  assert.strictEqual(clusters.length, 2, 'Should create 2 clusters');
  
  // Verify cluster sizes
  assert.strictEqual(clusters[0].embeddings.length, 3, 'First cluster should have 3 embeddings');
  assert.strictEqual(clusters[1].embeddings.length, 3, 'Second cluster should have 3 embeddings');
  
  // Verify centroids are calculated correctly
  const expectedCentroid1 = [0.9, 0.1];
  const expectedCentroid2 = [0.1, 0.9];
  
  // Check similarity between actual and expected centroids
  const similarity1 = cosineSimilarity(clusters[0].centroid, expectedCentroid1);
  const similarity2 = cosineSimilarity(clusters[1].centroid, expectedCentroid2);
  
  assert(similarity1 > 0.99, 'First centroid should be close to expected');
  assert(similarity2 > 0.99, 'Second centroid should be close to expected');
  
  console.log('✓ Basic clustering test passed');
}

// Test 2: Clustering with phrases
function testClusteringWithPhrases() {
  console.log('\nTest 2: Clustering with phrases');
  
  // Create test data with phrases
  const embeddings = [
    [1.0, 0.0],
    [0.9, 0.1],
    [0.0, 1.0],
    [0.1, 0.9],
  ];
  
  const phrasesWithEmbeddings = [
    { phrase: "This is phrase 1", embedding: embeddings[0] },
    { phrase: "This is phrase 2", embedding: embeddings[1] },
    { phrase: "This is phrase 3", embedding: embeddings[2] },
    { phrase: "This is phrase 4", embedding: embeddings[3] },
  ];
  
  // Configure clustering for testing
  updateClusteringConfig({
    enableClustering: true,
    similarityThreshold: 0.9,
    minClusterSize: 1,
    maxClusters: 2
  });
  
  // Run clustering
  const clusters = clusterEmbeddings(embeddings, phrasesWithEmbeddings);
  
  // Verify we have 2 clusters
  assert.strictEqual(clusters.length, 2, 'Should create 2 clusters');
  
  // Verify phrases are assigned to correct clusters
  assert.strictEqual(clusters[0].phrases.length, 2, 'First cluster should have 2 phrases');
  assert.strictEqual(clusters[1].phrases.length, 2, 'Second cluster should have 2 phrases');
  
  // Check that phrases are correctly assigned
  assert(clusters[0].phrases.includes("This is phrase 1"), 'Cluster 1 should contain phrase 1');
  assert(clusters[0].phrases.includes("This is phrase 2"), 'Cluster 1 should contain phrase 2');
  assert(clusters[1].phrases.includes("This is phrase 3"), 'Cluster 2 should contain phrase 3');
  assert(clusters[1].phrases.includes("This is phrase 4"), 'Cluster 2 should contain phrase 4');
  
  console.log('✓ Clustering with phrases test passed');
}

// Test 3: Disabled clustering
function testDisabledClustering() {
  console.log('\nTest 3: Disabled clustering');
  
  // Create some test embeddings
  const embeddings = [
    [1.0, 0.0],
    [0.9, 0.1],
    [0.0, 1.0],
    [0.1, 0.9],
  ];
  
  // Configure clustering to be disabled
  updateClusteringConfig({
    enableClustering: false
  });
  
  // Run clustering
  const clusters = clusterEmbeddings(embeddings);
  
  // Verify we have only 1 cluster
  assert.strictEqual(clusters.length, 1, 'Should create only 1 cluster when disabled');
  
  // Verify all embeddings are in the single cluster
  assert.strictEqual(clusters[0].embeddings.length, 4, 'Single cluster should contain all embeddings');
  
  console.log('✓ Disabled clustering test passed');
}

// Test 4: Small clusters handling
function testSmallClustersHandling() {
  console.log('\nTest 4: Small clusters handling');
  
  // Create test embeddings with 3 potential clusters
  const embeddings = [
    [1.0, 0.0],  // Cluster 1 (3 items)
    [0.9, 0.1],
    [0.95, 0.05],
    [0.0, 1.0],  // Cluster 2 (3 items)
    [0.1, 0.9],
    [0.05, 0.95],
    [0.5, 0.5],  // Cluster 3 (1 item) - too small, should be merged
  ];
  
  // Configure clustering to require min size of 2
  updateClusteringConfig({
    enableClustering: true,
    similarityThreshold: 0.9,
    minClusterSize: 2,
    maxClusters: 3
  });
  
  // Run clustering
  const clusters = clusterEmbeddings(embeddings);
  
  // Verify we have 2 clusters (the small one should be merged)
  assert.strictEqual(clusters.length, 2, 'Should merge small clusters');
  
  // Verify all embeddings are accounted for
  const totalEmbeddings = clusters.reduce((sum, cluster) => sum + cluster.embeddings.length, 0);
  assert.strictEqual(totalEmbeddings, 7, 'All embeddings should be assigned to clusters');
  
  console.log('✓ Small clusters handling test passed');
}

// Test 5: Average embedding calculation
function testAverageEmbedding() {
  console.log('\nTest 5: Average embedding calculation');
  
  // Test vectors
  const vectors = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
  ];
  
  // Expected average: [4, 5, 6]
  const average = calculateAverageEmbedding(vectors);
  
  // Check result
  assert.deepStrictEqual(average, [4, 5, 6], 'Average should be calculated correctly');
  
  console.log('✓ Average embedding calculation test passed');
}

// Test 6: Cohesion calculation
function testCohesionCalculation() {
  console.log('\nTest 6: Cohesion calculation');
  
  // Create test embeddings with known similarity
  const embeddings = [
    [1.0, 0.0],  // Perfect similarity with centroid
    [0.9, 0.1],  // High similarity
    [0.8, 0.2],  // Medium-high similarity
  ];
  
  const centroid = [1.0, 0.0];
  
  // Calculate expected cohesion
  const expectedCohesion = calculateCohesion(embeddings, centroid);
  
  // Verify the cohesion is calculated correctly
  // The cohesion should be the average of:
  // 1.0 (perfect similarity)
  // ~0.9 (high similarity)
  // ~0.8 (medium-high similarity)
  assert(expectedCohesion > 0.8 && expectedCohesion < 1.0, 'Cohesion should be between 0.8 and 1.0');
  
  // Test with a more diverse cluster
  const diverseEmbeddings = [
    [1.0, 0.0],   // Perfect similarity with centroid
    [0.7, 0.3],   // Medium similarity
    [0.5, 0.5],   // Lower similarity
  ];
  
  const diverseCohesion = calculateCohesion(diverseEmbeddings, centroid);
  
  // Verify the diverse cohesion is lower
  assert(diverseCohesion < expectedCohesion, 'Diverse cluster should have lower cohesion');
  assert(diverseCohesion > 0.6 && diverseCohesion < 0.9, 'Diverse cohesion should be between 0.6 and 0.9');
  
  console.log('✓ Cohesion calculation test passed');
}

// Run all tests
try {
  testBasicClustering();
  testClusteringWithPhrases();
  testDisabledClustering();
  testSmallClustersHandling();
  testAverageEmbedding();
  testCohesionCalculation();
  
  console.log('\n✅ All clustering tests passed!');
} catch (error) {
  console.error('\n❌ Test failed:', error);
  process.exit(1);
} 