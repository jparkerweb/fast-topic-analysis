// -----------------------------------------------------
// -- Clustering functionality for topic embeddings --
// -----------------------------------------------------
import { cosineSimilarity } from './similarity.js';
import { toBoolean } from './utils.js';
import dotenv from 'dotenv';
dotenv.config();

// --------------------------
// -- Clustering settings --
// --------------------------
export const clusteringConfig = {
  enabled: toBoolean(process.env.ENABLE_CLUSTERING) ?? true,
  similarityThreshold: parseFloat(process.env.CLUSTERING_SIMILARITY_THRESHOLD) || 0.9,
  minClusterSize: parseInt(process.env.CLUSTERING_MIN_CLUSTER_SIZE) || 5,
  maxClusters: parseInt(process.env.CLUSTERING_MAX_CLUSTERS) || 5
};

// Update clustering config with command line arguments
export function updateClusteringConfig(args = {}) {
  if (args.enableClustering !== undefined) {
    clusteringConfig.enabled = toBoolean(args.enableClustering);
  }
  
  if (args.similarityThreshold !== undefined) {
    clusteringConfig.similarityThreshold = parseFloat(args.similarityThreshold);
  }
  
  if (args.minClusterSize !== undefined) {
    clusteringConfig.minClusterSize = parseInt(args.minClusterSize);
  }
  
  if (args.maxClusters !== undefined) {
    clusteringConfig.maxClusters = parseInt(args.maxClusters);
  }
  
  // Apply presets if specified
  if (args.preset) {
    applyPreset(args.preset);
  }
  
  return clusteringConfig;
}

// Apply configuration presets
function applyPreset(preset) {
  switch (preset.toLowerCase()) {
    case 'high-precision':
      clusteringConfig.enabled = true;
      clusteringConfig.similarityThreshold = 0.95;
      clusteringConfig.minClusterSize = 3;
      clusteringConfig.maxClusters = 8;
      break;
    case 'balanced':
      clusteringConfig.enabled = true;
      clusteringConfig.similarityThreshold = 0.9;
      clusteringConfig.minClusterSize = 5;
      clusteringConfig.maxClusters = 5;
      break;
    case 'performance':
      clusteringConfig.enabled = true;
      clusteringConfig.similarityThreshold = 0.85;
      clusteringConfig.minClusterSize = 10;
      clusteringConfig.maxClusters = 3;
      break;
    case 'legacy':
      clusteringConfig.enabled = false;
      break;
    default:
      console.warn(`Unknown preset: ${preset}. Using current configuration.`);
  }
}

// -----------------------------------------------------
// -- Calculate average embedding for a set of vectors --
// -----------------------------------------------------
export function calculateAverageEmbedding(embeddings) {
  if (!embeddings || embeddings.length === 0) {
    throw new Error('Cannot calculate average of empty embeddings array');
  }
  
  const numEmbeddings = embeddings.length;
  const embeddingSize = embeddings[0].length;
  const averagedEmbedding = new Array(embeddingSize).fill(0);
  
  for (let i = 0; i < numEmbeddings; i++) {
    for (let j = 0; j < embeddingSize; j++) {
      averagedEmbedding[j] += embeddings[i][j];
    }
  }
  
  for (let j = 0; j < embeddingSize; j++) {
    averagedEmbedding[j] /= numEmbeddings;
  }
  
  return averagedEmbedding;
}

// -------------------------------------------------------
// -- Calculate similarity between embedding and cluster --
// -------------------------------------------------------
function calculateSimilarityToCluster(embedding, cluster) {
  if (cluster.length === 0) return 0;
  
  let totalSimilarity = 0;
  for (const clusterEmbedding of cluster) {
    totalSimilarity += cosineSimilarity(embedding, clusterEmbedding);
  }
  
  return totalSimilarity / cluster.length;
}

// -------------------------------------------------------
// -- Find most similar cluster for a given embedding --
// -------------------------------------------------------
function findMostSimilarCluster(embedding, clusters) {
  let maxSimilarity = -1;
  let mostSimilarClusterIndex = -1;
  
  for (let i = 0; i < clusters.length; i++) {
    const similarity = calculateSimilarityToCluster(embedding, clusters[i]);
    if (similarity > maxSimilarity) {
      maxSimilarity = similarity;
      mostSimilarClusterIndex = i;
    }
  }
  
  return { 
    clusterIndex: mostSimilarClusterIndex, 
    similarity: maxSimilarity 
  };
}

// -------------------------------------------------------
// -- Cluster embeddings using similarity-based approach --
// -------------------------------------------------------
export function clusterEmbeddings(embeddings, phrasesWithEmbeddings = null) {
  if (!clusteringConfig.enabled) {
    // If clustering is disabled, return a single cluster with all embeddings
    return [{
      embeddings: embeddings,
      phrases: phrasesWithEmbeddings ? phrasesWithEmbeddings.map(item => item.phrase) : [],
      centroid: calculateAverageEmbedding(embeddings)
    }];
  }
  
  const { similarityThreshold, minClusterSize, maxClusters } = clusteringConfig;
  
  // Initialize clusters with the first embedding
  const clusters = [[embeddings[0]]];
  const clusterPhrases = phrasesWithEmbeddings ? [[phrasesWithEmbeddings[0].phrase]] : [[]];
  
  // Assign each embedding to a cluster
  for (let i = 1; i < embeddings.length; i++) {
    const embedding = embeddings[i];
    const { clusterIndex, similarity } = findMostSimilarCluster(embedding, clusters);
    
    // If similarity is above threshold and we have a valid cluster, add to that cluster
    if (similarity >= similarityThreshold && clusterIndex !== -1) {
      clusters[clusterIndex].push(embedding);
      if (phrasesWithEmbeddings) {
        clusterPhrases[clusterIndex].push(phrasesWithEmbeddings[i].phrase);
      }
    } 
    // If we haven't reached max clusters, create a new one
    else if (clusters.length < maxClusters) {
      clusters.push([embedding]);
      if (phrasesWithEmbeddings) {
        clusterPhrases.push([phrasesWithEmbeddings[i].phrase]);
      }
    } 
    // Otherwise, add to the most similar cluster
    else {
      clusters[clusterIndex].push(embedding);
      if (phrasesWithEmbeddings) {
        clusterPhrases[clusterIndex].push(phrasesWithEmbeddings[i].phrase);
      }
    }
  }
  
  // Handle small clusters
  const validClusters = [];
  const validClusterPhrases = [];
  const smallClusters = [];
  const smallClusterPhrases = [];
  
  // Separate valid and small clusters
  for (let i = 0; i < clusters.length; i++) {
    if (clusters[i].length >= minClusterSize) {
      validClusters.push(clusters[i]);
      if (phrasesWithEmbeddings) {
        validClusterPhrases.push(clusterPhrases[i]);
      }
    } else {
      smallClusters.push(...clusters[i]);
      if (phrasesWithEmbeddings) {
        smallClusterPhrases.push(...clusterPhrases[i]);
      }
    }
  }
  
  // If we have small clusters and valid clusters, distribute them
  if (smallClusters.length > 0 && validClusters.length > 0) {
    for (let i = 0; i < smallClusters.length; i++) {
      const embedding = smallClusters[i];
      const { clusterIndex } = findMostSimilarCluster(embedding, validClusters);
      
      validClusters[clusterIndex].push(embedding);
      if (phrasesWithEmbeddings) {
        validClusterPhrases[clusterIndex].push(smallClusterPhrases[i]);
      }
    }
  } 
  // If all clusters are small or we have no valid clusters, create a single "miscellaneous" cluster
  else if (smallClusters.length > 0) {
    validClusters.push(smallClusters);
    if (phrasesWithEmbeddings) {
      validClusterPhrases.push(smallClusterPhrases);
    }
  }
  
  // Calculate centroids for each cluster
  const result = validClusters.map((cluster, index) => {
    return {
      embeddings: cluster,
      phrases: phrasesWithEmbeddings ? validClusterPhrases[index] : [],
      centroid: calculateAverageEmbedding(cluster)
    };
  });
  
  return result;
} 