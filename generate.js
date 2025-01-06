// -------------
// -- imports --
// -------------
import { combineTopicEmbeddings } from "./modules/embedding.js";
import { labels } from "./labels-config.js";
import fs from 'fs';

console.log('\n\n\n\n');

// ------------------------------------------
// -- Clean the topic_embeddings directory --
// ------------------------------------------
const topicEmbeddingsDir = 'data/topic_embeddings';
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
    const existingEmbedding = null;
    const existingCount = 0;
    const topicName = label.label;
    const threshold = label.threshold;
    
    const newPhrases = allTrainPositivesArray
        .filter(item => item.label.toLowerCase() === topicName.toLowerCase())
        .map(item => item.text);

    if (newPhrases.length === 0) {
        console.log(`No training data found for topic "${topicName}" - skipping embedding generation`);
        return;
    }

    let topicEmbedding;
    try {
        topicEmbedding = await combineTopicEmbeddings(existingEmbedding, existingCount, newPhrases);
        const dataObject = {
            topic: topicName,
            threshold: threshold,
            numPhrases: newPhrases.length,
            embeddingModel: process.env.ONNX_EMBEDDING_MODEL,
            modelPrecision: process.env.ONNX_EMBEDDING_MODEL_PRECISION,
            embedding: Array.isArray(topicEmbedding) ? topicEmbedding : Object.values(topicEmbedding)
        };
        const dataString = JSON.stringify(dataObject, null, 2);
        fs.writeFileSync(`data/topic_embeddings/${topicName}.json`, dataString, { flag: 'w' });
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
