// -------------
// -- imports --
// -------------
import { combineTopicEmbeddings, generateEmbeddings, prefixConfig, weightedAverage } from "./modules/embedding.js";
import { clusterEmbeddings, updateClusteringConfig, clusteringConfig } from "./modules/clusterEmbeddings.js";
import { cosineSimilarity } from "./modules/similarity.js";
import { labels } from "./labels-config.js";
import { loadManifest, validateManifest, getNewLines, updateManifest, createManifest } from './modules/manifest.js';
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

// -------------------------------------------
// -- Manifest path constant               --
// -------------------------------------------
const MANIFEST_PATH = 'data/incremental-manifest.json';

// ----------------------------
// -- Incremental mode check --
// ----------------------------
if (args.incremental) {
  await incrementalGenerate();
  process.exit(0);
}

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


// -------------------------------------------
// -- Incremental generation function       --
// -------------------------------------------

async function incrementalGenerate() {
  // Task 2.5: Load and validate manifest
  const manifest = loadManifest(MANIFEST_PATH);
  if (!manifest) {
    console.log('No manifest found. Run full generation first: node generate.js');
    process.exit(1);
  }

  const validation = validateManifest(manifest, 'data/training_data.jsonl');
  if (!validation.valid) {
    const messages = {
      hash_mismatch: 'Training data has been modified. Run full generation: node generate.js',
      model_mismatch: 'Embedding model changed. Run full generation: node generate.js',
      precision_mismatch: 'Model precision changed. Run full generation: node generate.js'
    };
    console.log(messages[validation.reason]);
    process.exit(1);
  }

  const newEntries = getNewLines('data/training_data.jsonl', manifest.lastProcessedLine);
  if (newEntries.length === 0) {
    console.log('No new training data found. Nothing to update.');
    return;
  }

  // Task 2.6: Group new lines by topic and validate
  const topicGroups = new Map();
  for (const entry of newEntries) {
    const topicLabel = entry.label.toLowerCase();
    if (!topicGroups.has(topicLabel)) {
      topicGroups.set(topicLabel, []);
    }
    topicGroups.get(topicLabel).push(entry);
  }

  const topicEmbeddingsDir = 'data/topic_embeddings';
  for (const [topicLabel] of topicGroups) {
    const clusterFiles = fs.readdirSync(topicEmbeddingsDir)
      .filter(file => file.startsWith(`${topicLabel}-cluster-`) && file.endsWith('.json'));
    if (clusterFiles.length === 0) {
      console.log(`New topic '${topicLabel}' found with no existing clusters. Run full generation: node generate.js`);
      process.exit(1);
    }
  }

  // Task 2.7: Embed new phrases, assign to nearest cluster, update centroids
  for (const [topicLabel, entries] of topicGroups) {
    const phrases = entries.map(e => e.text);
    const embeddings = await generateEmbeddings(phrases, {
      prefix: prefixConfig.dataPrefix,
      returnPhrases: false,
      logging: false
    });

    // Load all cluster files for this topic
    const clusterFiles = fs.readdirSync(topicEmbeddingsDir)
      .filter(file => file.startsWith(`${topicLabel}-cluster-`) && file.endsWith('.json'));
    const clusters = clusterFiles.map(file => ({
      filename: file,
      data: JSON.parse(fs.readFileSync(path.join(topicEmbeddingsDir, file), 'utf8'))
    }));

    // Assign each new embedding to nearest cluster and update centroid
    for (const newEmbedding of embeddings) {
      let bestIdx = 0;
      let bestSim = -Infinity;
      for (let i = 0; i < clusters.length; i++) {
        const sim = cosineSimilarity(Array.from(newEmbedding), clusters[i].data.embedding);
        if (sim > bestSim) {
          bestSim = sim;
          bestIdx = i;
        }
      }
      clusters[bestIdx].data.embedding = weightedAverage(
        clusters[bestIdx].data.embedding,
        clusters[bestIdx].data.clusterSize,
        [Array.from(newEmbedding)]
      );
      clusters[bestIdx].data.clusterSize += 1;
    }

    // Update totalPhrases on all clusters for this topic
    for (const cluster of clusters) {
      cluster.data.totalPhrases = cluster.data.totalPhrases + phrases.length;
      fs.writeFileSync(
        path.join(topicEmbeddingsDir, cluster.filename),
        JSON.stringify(cluster.data, null, 2)
      );
    }

    console.log(`Topic '${topicLabel}': updated ${clusterFiles.length} clusters with ${phrases.length} new phrases`);
  }

  // Task 2.8: Update manifest
  const totalValidLines = manifest.lastProcessedLine + newEntries.length;
  updateManifest(MANIFEST_PATH, totalValidLines, 'data/training_data.jsonl', process.env.ONNX_EMBEDDING_MODEL, process.env.ONNX_EMBEDDING_MODEL_PRECISION);
  console.log('Incremental update complete. Manifest updated.');
}

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
    await generateTopicEmbedding(label);
}

// -- Create manifest after full generation --
createManifest('data/training_data.jsonl', MANIFEST_PATH, process.env.ONNX_EMBEDDING_MODEL, process.env.ONNX_EMBEDDING_MODEL_PRECISION);
console.log('Manifest created. Ready for incremental updates.');

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
    } else if (arg === '--incremental') {
      args.incremental = true;
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
  --incremental              Update clusters incrementally (new JSONL entries only)
  --help                      Show this help message

Examples:
  node generate.js --preset high-precision
  node generate.js --enable-clustering true --similarity-threshold 0.92
  node generate.js --max-clusters 3
  node generate.js --incremental
`);
}
