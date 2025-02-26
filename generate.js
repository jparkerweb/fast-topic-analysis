// -------------
// -- imports --
// -------------
import { combineTopicEmbeddings, generateEmbeddings, prefixConfig } from "./modules/embedding.js";
import { clusterEmbeddings, updateClusteringConfig, clusteringConfig } from "./modules/clusterEmbeddings.js";
import { cosineSimilarity } from "./modules/similarity.js";
import { labels } from "./labels-config.js";
import fs from 'fs';
import path from 'path';

// Parse command line arguments
const args = parseCommandLineArgs();

// Update clustering configuration if needed
if (Object.keys(args).length > 0) {
  updateClusteringConfig(args);
  console.log('Clustering configuration:', clusteringConfig);
}

console.log('\n\n\n\n');

// ------------------------------------------
// -- Clean the topic_embeddings directory --
// ------------------------------------------
const topicEmbeddingsDir = 'data/topic_embeddings';
if (!fs.existsSync(topicEmbeddingsDir)) {
  fs.mkdirSync(topicEmbeddingsDir, { recursive: true });
}

fs.readdirSync(topicEmbeddingsDir)
    .filter(file => file.endsWith('.json'))
    .forEach(file => {
        fs.unlinkSync(`${topicEmbeddingsDir}/${file}`);
        console.log(`Deleted: ${file}`);
    });
console.log('\nCleaned topic_embeddings directory\n');


// ---------------------------------------------------------------------------------
// -- Load `data/training_data.jsonl` and get all the phrases for the label --
// ---------------------------------------------------------------------------------
const allTrainPositives = fs.readFileSync('data/training_data.jsonl', 'utf8');
const allTrainPositivesArray = allTrainPositives.split('\n')
    .map(line => {
        try {
            return JSON.parse(line);
        } catch (e) {
            // console.error('Failed to parse JSON:', line.slice(0, 100) + '...');
            return null; // Return null instead of undefined
        }
    })
    .filter(item => item !== null); // Remove null entries before processing


// ---------------------------------------------------
// -- Generate the topic average weighted embedding --
// ---------------------------------------------------
async function generateTopicEmbedding(label) {
    const topicName = label.label;
    const threshold = label.threshold;
    
    const newPhrases = allTrainPositivesArray
        .filter(item => item.label.toLowerCase() === topicName.toLowerCase())
        .map(item => item.text);

    if (newPhrases.length === 0) {
        console.log(`No training data found for topic "${topicName}" - skipping embedding generation`);
        return;
    }

    try {
        // Generate embeddings for all phrases in the topic
        const phrasesWithEmbeddings = await generateEmbeddings(newPhrases, {
            prefix: prefixConfig.dataPrefix,
            returnPhrases: true,
            logging: false,
        });
        
        // Extract just the embeddings for clustering
        const embeddings = phrasesWithEmbeddings.map(item => item.embedding);
        
        // Cluster the embeddings
        const clusters = clusterEmbeddings(embeddings, phrasesWithEmbeddings);
        
        console.log(`Topic "${topicName}" generated ${clusters.length} clusters`);
        
        // Save each cluster as a separate embedding file
        for (let i = 0; i < clusters.length; i++) {
            const cluster = clusters[i];
            const clusterSize = cluster.embeddings.length;
            const clusterCoverage = (clusterSize / newPhrases.length * 100).toFixed(2);
            
            // Calculate cohesion - average similarity between all embeddings and the centroid
            let totalSimilarity = 0;
            for (const embedding of cluster.embeddings) {
                totalSimilarity += cosineSimilarity(embedding, cluster.centroid);
            }
            const cohesion = clusterSize > 0 ? totalSimilarity / clusterSize : 1.0;
            
            const dataObject = {
                topic: topicName,
                threshold: threshold,
                clusterIndex: i,
                totalClusters: clusters.length,
                clusterSize: clusterSize,
                clusterCoverage: `${clusterCoverage}%`,
                cohesion: cohesion.toFixed(4),
                totalPhrases: newPhrases.length,
                embeddingModel: process.env.ONNX_EMBEDDING_MODEL,
                modelPrecision: process.env.ONNX_EMBEDDING_MODEL_PRECISION,
                embedding: cluster.centroid
            };
            
            // Create filename: topic-cluster-X-of-Y.json
            const filename = `${topicName}-cluster-${i+1}-of-${clusters.length}.json`;
            const dataString = JSON.stringify(dataObject, null, 2);
            fs.writeFileSync(path.join(topicEmbeddingsDir, filename), dataString, { flag: 'w' });
            
            console.log(`  - Cluster ${i+1}/${clusters.length}: ${clusterSize} phrases (${clusterCoverage}% coverage, cohesion: ${cohesion.toFixed(4)})`);
        }
        
        console.log(`Topic embedding for ${topicName} generated successfully`);
    } catch (error) {
        console.error(`Error generating topic embedding for "${topicName}":`, error);
        process.exit(1);
    }
}


// ------------------------------------------------------------------
// -- Loop through labels and generate average weighted embeddings --
// ------------------------------------------------------------------
for (const label of labels) {
    generateTopicEmbedding(label);
}

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
Usage: node generate.js [options]

Options:
  --preset, -p <name>         Use a predefined configuration preset
                              (high-precision, balanced, performance, legacy)
  --enable-clustering <bool>  Enable or disable clustering (true/false)
  --similarity-threshold <n>  Set similarity threshold for clustering (0-1)
  --min-cluster-size <n>      Set minimum cluster size
  --max-clusters <n>          Set maximum number of clusters per topic
  --help                      Show this help message

Examples:
  node generate.js --preset high-precision
  node generate.js --enable-clustering true --similarity-threshold 0.92
  node generate.js --max-clusters 3
`);
}
