// -------------------------------------------------
// -- import environment variables from .env file --
// -------------------------------------------------
import dotenv from 'dotenv';
dotenv.config();

import { createLocalProvider, batchIncrementalAverage, averageEmbeddings } from 'embedding-utils';
import { toBoolean } from './utils.js';

// ---------------------
// -- model variables --
// ---------------------
const ONNX_EMBEDDING_MODEL = process.env.ONNX_EMBEDDING_MODEL || 'Xenova/all-MiniLM-L6-v2';
const ONNX_EMBEDDING_MODEL_PRECISION = process.env.ONNX_EMBEDDING_MODEL_PRECISION || 'fp32';
console.log(`Using embedding model: ${ONNX_EMBEDDING_MODEL} with precision: ${ONNX_EMBEDDING_MODEL_PRECISION}`);

// ---------------------------------------
// -- Initialize the embedding provider --
// -- Prefixes are auto-detected by    --
// -- embedding-utils from its registry. --
// -- Pass documentPrefix/queryPrefix    --
// -- only to override the default.      --
// ---------------------------------------
export const provider = createLocalProvider({
    model: ONNX_EMBEDDING_MODEL,
    precision: ONNX_EMBEDDING_MODEL_PRECISION,
    documentPrefix: process.env.ONNX_EMBEDDING_MODEL_DOCUMENT_PREFIX,
    queryPrefix: process.env.ONNX_EMBEDDING_MODEL_QUERY_PREFIX,
    modelPath: process.env.LOCAL_MODEL_PATH,
    cacheDir: process.env.CACHE_DIR,
    allowRemoteModels: process.env.ALLOW_REMOTE_MODELS !== undefined ? toBoolean(process.env.ALLOW_REMOTE_MODELS) : true,
});

// -----------------------------------------------------
// -- Function to generate embeddings for new phrases --
// -- inputType: 'document' (training data) or 'query' --
// -- (analysis text). embedding-utils auto-applies    --
// -- the correct prefix from its model registry.      --
// -----------------------------------------------------
export async function generateEmbeddings(phrases, { inputType = 'document', returnPhrases = false, logging = false } = {}) {
    const { embeddings } = await provider.embed(phrases, { inputType });

    return phrases.map((phrase, i) => {
        const embedding = Array.from(embeddings[i]);
        if (logging) { console.log(`Generated Embedding for phrase "${phrase}":`); }
        return returnPhrases ? { phrase, embedding } : embedding;
    });
}

// ------------------------------------------------------------
// -- Main function to combine embeddings for topic creation --
// ------------------------------------------------------------
export async function combineTopicEmbeddings(existingEmbedding, existingCount, newPhrases) {
    const newEmbeddings = await generateEmbeddings(newPhrases, {
        inputType: 'document',
        returnPhrases: false,
        logging: false,
    });

    if (!existingEmbedding) {
        return Array.from(averageEmbeddings(newEmbeddings));
    }

    return Array.from(batchIncrementalAverage(existingEmbedding, newEmbeddings, existingCount));
}
