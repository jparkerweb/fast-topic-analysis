// -------------
// -- imports --
// -------------
import { generateEmbeddings, prefixConfig } from "../modules/embedding.js";
import { clusterEmbeddings, updateClusteringConfig } from "../modules/clusterEmbeddings.js";
import { cosineSimilarity } from "../modules/similarity.js";
import fs from 'fs';
import chalk from 'chalk';

// Parse command line arguments
const args = parseCommandLineArgs();

// Update clustering configuration if needed
if (Object.keys(args).length > 0) {
  updateClusteringConfig(args);
}

// Calculate cohesion for a cluster
function calculateClusterCohesion(cluster) {
  let totalSimilarity = 0;
  for (const embedding of cluster.embeddings) {
    totalSimilarity += cosineSimilarity(embedding, cluster.centroid);
  }
  return cluster.embeddings.length > 0 ? totalSimilarity / cluster.embeddings.length : 1.0;
}

// Load test data
async function runClusteringTest() {
  console.log(chalk.blue('\n=== Clustering Test ===\n'));
  
  // Load training data
  const allTrainPositives = fs.readFileSync('data/training_data.jsonl', 'utf8');
  const allTrainPositivesArray = allTrainPositives.split('\n')
    .map(line => {
      try {
        return JSON.parse(line);
      } catch (e) {
        return null;
      }
    })
    .filter(item => item !== null);
  
  // Get unique topics
  const topics = [...new Set(allTrainPositivesArray.map(item => item.label.toLowerCase()))];
  
  // Test clustering for each topic
  for (const topic of topics) {
    console.log(chalk.green(`\nTesting clustering for topic: ${topic}`));
    
    // Get phrases for this topic
    const topicPhrases = allTrainPositivesArray
      .filter(item => item.label.toLowerCase() === topic)
      .map(item => item.text);
    
    console.log(`Found ${topicPhrases.length} phrases for topic "${topic}"`);
    
    // Generate embeddings
    const phrasesWithEmbeddings = await generateEmbeddings(topicPhrases, {
      prefix: prefixConfig.dataPrefix,
      returnPhrases: true,
      logging: false,
    });
    
    // Extract just the embeddings for clustering
    const embeddings = phrasesWithEmbeddings.map(item => item.embedding);
    
    // Cluster the embeddings
    const clusters = clusterEmbeddings(embeddings, phrasesWithEmbeddings);
    
    // Print cluster information
    console.log(`Generated ${clusters.length} clusters:`);
    
    for (let i = 0; i < clusters.length; i++) {
      const cluster = clusters[i];
      const clusterSize = cluster.embeddings.length;
      const clusterCoverage = (clusterSize / topicPhrases.length * 100).toFixed(2);
      const cohesion = calculateClusterCohesion(cluster).toFixed(4);
      
      console.log(`  - Cluster ${i+1}/${clusters.length}: ${clusterSize} phrases (${clusterCoverage}% coverage, cohesion: ${cohesion})`);
      
      // Print a few example phrases from this cluster
      const exampleCount = Math.min(3, cluster.phrases.length);
      console.log(`    Example phrases:`);
      for (let j = 0; j < exampleCount; j++) {
        console.log(`      - "${cluster.phrases[j]}"`);
      }
    }
  }
}

// Run the test
runClusteringTest().catch(error => {
  console.error('Error running clustering test:', error);
  process.exit(1);
});

// ----------------------------------
// -- Parse command line arguments --
// ----------------------------------
function parseCommandLineArgs() {
  const args = {};
  const argv = process.argv.slice(2);
  
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    
    if (arg === '--preset' || arg === '-p') {
      args.preset = argv[++i];
    } else if (arg === '--enable-clustering') {
      args.enableClustering = argv[++i];
    } else if (arg === '--similarity-threshold') {
      args.similarityThreshold = argv[++i];
    } else if (arg === '--min-cluster-size') {
      args.minClusterSize = argv[++i];
    } else if (arg === '--max-clusters') {
      args.maxClusters = argv[++i];
    } else if (arg === '--help') {
      printHelp();
      process.exit(0);
    }
  }
  
  return args;
}

// -----------------------
// -- Print help message --
// -----------------------
function printHelp() {
  console.log(`
Usage: node test-clustering.js [options]

Options:
  --preset, -p <name>         Use a predefined configuration preset
                              (high-precision, balanced, performance, legacy)
  --enable-clustering <bool>  Enable or disable clustering (true/false)
  --similarity-threshold <n>  Set similarity threshold for clustering (0-1)
  --min-cluster-size <n>      Set minimum cluster size
  --max-clusters <n>          Set maximum number of clusters per topic
  --help                      Show this help message

Examples:
  node test-clustering.js --preset high-precision
  node test-clustering.js --similarity-threshold 0.92
  node test-clustering.js --max-clusters 3
`);
} 