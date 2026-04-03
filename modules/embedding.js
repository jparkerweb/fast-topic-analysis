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
const requiresPrefix = toBoolean(process.env.ONNX_EMBEDDING_MODEL_REQUIRES_PREFIX) || false;

export const prefixConfig = {
    dataPrefix: requiresPrefix ? (process.env.ONNX_EMBEDDING_MODEL_DOCUMENT_PREFIX || 'search_document: ') : '',
    queryPrefix: requiresPrefix ? (process.env.ONNX_EMBEDDING_MODEL_QUERY_PREFIX || 'search_query: ') : '',
};

const ONNX_EMBEDDING_MODEL = process.env.ONNX_EMBEDDING_MODEL || 'all-MiniLM-L6-v2';
const ONNX_EMBEDDING_MODEL_PRECISION = process.env.ONNX_EMBEDDING_MODEL_PRECISION || 'fp32';
console.log(`Using embedding model: ${ONNX_EMBEDDING_MODEL} with precision: ${ONNX_EMBEDDING_MODEL_PRECISION}`);

// ---------------------------------------
// -- Initialize the embedding provider --
// ---------------------------------------
export const provider = createLocalProvider({
    model: ONNX_EMBEDDING_MODEL,
    precision: ONNX_EMBEDDING_MODEL_PRECISION,
    documentPrefix: requiresPrefix ? (process.env.ONNX_EMBEDDING_MODEL_DOCUMENT_PREFIX || 'search_document: ') : '',
    queryPrefix: requiresPrefix ? (process.env.ONNX_EMBEDDING_MODEL_QUERY_PREFIX || 'search_query: ') : '',
    modelPath: process.env.LOCAL_MODEL_PATH,
    cacheDir: process.env.CACHE_DIR,
    allowRemoteModels: toBoolean(process.env.ALLOW_REMOTE_MODELS) || true,
});

// -----------------------------------------------------
// -- Function to generate embeddings for new phrases --
// -----------------------------------------------------
export async function generateEmbeddings(phrases, { prefix = '', returnPhrases = false, logging = false } = {}) {
    const prefixedPhrases = phrases.map(phrase => `${prefix}${phrase}`);
    const { embeddings } = await provider.embed(prefixedPhrases);

    return phrases.map((phrase, i) => {
        const embedding = Array.from(embeddings[i]);
        if (logging) { console.log(`Generated Embedding for phrase "${prefix}${phrase}":`); }
        return returnPhrases ? { phrase: `${prefix}${phrase}`, embedding } : embedding;
    });
}

// ------------------------------------------------------------
// -- Main function to combine embeddings for topic creation --
// ------------------------------------------------------------
export async function combineTopicEmbeddings(existingEmbedding, existingCount, newPhrases) {
    const newEmbeddings = await generateEmbeddings(newPhrases, {
        prefix: prefixConfig.dataPrefix,
        returnPhrases: false,
        logging: false,
    });

    if (!existingEmbedding) {
        return averageEmbeddings(newEmbeddings);
    }

    const combinedEmbedding = batchIncrementalAverage(existingEmbedding, newEmbeddings, existingCount);
    return combinedEmbedding;
}
