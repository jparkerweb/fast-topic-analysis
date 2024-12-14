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

const ONNX_EMBEDDING_MODEL = process.env.ONNX_EMBEDDING_MODEL || 'all-MiniLM-L6-v2';
const ONNX_EMBEDDING_MODEL_PERCISION = process.env.ONNX_EMBEDDING_MODEL_PERCISION || 'fp32';

// ---------------------------------------
// -- Initialize the embedding pipeline --
// ---------------------------------------
const embeddingPipeline = await pipeline('feature-extraction', ONNX_EMBEDDING_MODEL, {
    dtype: ONNX_EMBEDDING_MODEL_PERCISION
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
export async function generateEmbeddings(phrases, returnPhrases = false) {
    const embeddings = await Promise.all(phrases.map(async (phrase) => {
      const tokens = await tokenizer(phrase, { truncation: true });
      if (tokens.input_ids.length > maxTokenLength) {
        console.warn(`Warning: The phrase "${phrase}" exceeds the maximum token length of ${maxTokenLength}. It will be truncated.`);
      }
      const result = await embeddingPipeline(phrase, { pooling: 'mean', normalize: true });
      // console.log(`Embedding for phrase "${phrase}":`, result); // Log the result
      return returnPhrases ? { phrase, embedding: result.data } : result.data;
    }));
    return embeddings;
}

// -----------------------------------------
// -- Main function to combine embeddings --
// -----------------------------------------
export async function combineEmbeddings(existingEmbedding, existingCount, newPhrases) {
  const newEmbeddings = await generateEmbeddings(newPhrases);
  
  if (!existingEmbedding) {
    return averageEmbeddings(newEmbeddings);
  }

  const combinedEmbedding = weightedAverage(existingEmbedding, existingCount, newEmbeddings);
  return combinedEmbedding;
}
