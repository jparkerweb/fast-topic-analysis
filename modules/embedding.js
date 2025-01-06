// -------------------------------------------------
// -- import environment variables from .env file --
// -------------------------------------------------
import dotenv from 'dotenv';
dotenv.config();

import { env, pipeline, AutoTokenizer } from '@huggingface/transformers';
import { toBoolean } from './utils.js';

// ---------------------
// -- model variables --
// ---------------------
env.localModelPath = process.env.LOCAL_MODEL_PATH; // local model path
env.cacheDir = process.env.CACHE_DIR;       // downloaded model cache directory
env.allowRemoteModels = toBoolean(process.env.ALLOW_REMOTE_MODELS) || true; // allow remote models (required for models to be be downloaded)
env.requiresPrefix = toBoolean(process.env.ONNX_EMBEDDING_MODEL_REQUIRES_PREFIX) || false; // require model prefix (nomic models rquire this)

export const prefixConfig = {
    dataPrefix: env.requiresPrefix ? (process.env.ONNX_EMBEDDING_MODEL_DOCUMENT_PREFIX || 'search_document: ') : '',
    queryPrefix: env.requiresPrefix ? (process.env.ONNX_EMBEDDING_MODEL_QUERY_PREFIX || 'search_query: ') : '',
};

const ONNX_EMBEDDING_MODEL = process.env.ONNX_EMBEDDING_MODEL || 'all-MiniLM-L6-v2';
const ONNX_EMBEDDING_MODEL_PRECISION = process.env.ONNX_EMBEDDING_MODEL_PRECISION || 'fp32';
console.log(`Using embedding model: ${ONNX_EMBEDDING_MODEL} with precision: ${ONNX_EMBEDDING_MODEL_PRECISION}`);

// ---------------------------------------
// -- Initialize the embedding pipeline --
// ---------------------------------------
const embeddingPipeline = await pipeline('feature-extraction', ONNX_EMBEDDING_MODEL, {
    dtype: ONNX_EMBEDDING_MODEL_PRECISION
});

// -------------------------------------------------
// -- Initialize the tokenizer and get max length --
// -------------------------------------------------
const tokenizer = await AutoTokenizer.from_pretrained(ONNX_EMBEDDING_MODEL);
const maxTokenLength = tokenizer.model_max_length;

// -----------------------------------------------------
// -- Function to calculate the average of embeddings --
// -----------------------------------------------------
function averageEmbeddings(embeddings) {
  const numEmbeddings = embeddings.length;
  const averagedEmbedding = embeddings[0].map((_, index) => {
    const sum = embeddings.reduce((acc, embedding) => acc + embedding[index], 0);
    return sum / numEmbeddings;
  });
  return averagedEmbedding;
}

// --------------------------------------------------------------
// -- Function to calculate the weighted average of embeddings --
// --------------------------------------------------------------
function weightedAverage(existingEmbedding, existingCount, newEmbeddings) {
  const totalMessages = existingCount + newEmbeddings.length;
  const weightedEmbedding = existingEmbedding.map((value, index) => {
    const newValue = newEmbeddings.reduce((acc, embedding) => acc + embedding[index], 0);
    return (value * existingCount + newValue) / totalMessages;
  });
  return weightedEmbedding;
}

// -----------------------------------------------------
// -- Function to generate embeddings for new phrases --
// -----------------------------------------------------
export async function generateEmbeddings(phrases, { prefix = '', returnPhrases = false, logging = false } = {}) {
    const embeddings = await Promise.all(phrases.map(async (phrase) => {
      phrase = `${prefix}${phrase}`;
      const tokens = await tokenizer(phrase, { truncation: true });
      if (tokens.input_ids.length > maxTokenLength) {
        console.warn(`Warning: The phrase "${phrase}" exceeds the maximum token length of ${maxTokenLength}. It will be truncated.`);
      }
      const result = await embeddingPipeline(phrase, { pooling: 'mean', normalize: true });
      if (logging) { console.log(`Generated Embedding for phrase "${phrase}":`); } // Log the phrase
      return returnPhrases ? { phrase, embedding: result.data } : result.data;
    }));
    return embeddings;
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

    const combinedEmbedding = weightedAverage(existingEmbedding, existingCount, newEmbeddings);
    return combinedEmbedding;
}
