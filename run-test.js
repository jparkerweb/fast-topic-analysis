// -------------
// -- imports --
// -------------
import { generateEmbeddings } from "./modules/embedding.js";
import { cosineSimilarity } from "./modules/similarity.js";
import { parseSentences } from 'sentence-parse';
import fs from 'fs';
import readline from 'readline';
import chalk from 'chalk';

// Create readline interface
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

// Load or create config
let config = { lastTestMessage: 1, verboseLogs: true, showMatches: true };
const configPath = './run-test-config.json';
try {
    config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
} catch {
    fs.writeFileSync(configPath, JSON.stringify(config, null, 4));
}

// Load topic embeddings from `data/topic_embeddings` folder
const topicEmbeddings = fs.readdirSync('data/topic_embeddings').map(file => {
    const topicName = file.split('.')[0];
    const topicEmbedding = fs.readFileSync(`data/topic_embeddings/${file}`, 'utf8');
    return { topicName, topicEmbedding: JSON.parse(topicEmbedding) };
});

// Load test message files
const testMessageFiles = fs.readdirSync('test-messages');

// Initialize sentence matches variable
let sentenceMatches = [];

// Initialize total comparisons variable
let totalComparisons = 0;

// Initialize total sentences variable
let totalSentences = 0;

// -------------------------------------------
// -- Test similarity for each test message --
// -------------------------------------------
async function testSimilarity(testMessage) {
    console.log(chalk.blue('\n-----------------------------------------------------\n'));

    const sentences = await parseSentences(testMessage);
    totalSentences += sentences.length;
    let sentencesWithEmbeddings = await generateEmbeddings(sentences, true);

    for (const { phrase, embedding } of sentencesWithEmbeddings) {
        if (config.verboseLogs) {
            console.log(`\nSentence: ${phrase}`);
        }
        
        for (const { topicName, topicEmbedding } of topicEmbeddings) {
            totalComparisons++;

            const similarity = cosineSimilarity(embedding, topicEmbedding.embedding);
            
            if (similarity >= topicEmbedding.threshold) {
                sentenceMatches.push({ topicName, phrase });

                if (config.verboseLogs) {
                    console.log(chalk.red(`Topic: ${topicName} ⇢ Similarity Score: ${similarity.toFixed(4)}`));
                } else if(!config.verboseLogs && config.showMatches) {
                    console.log(chalk.red(`Topic: ${topicName} ⇢ Similarity Score: ${similarity.toFixed(4)} ⇠ ${phrase}`));
                }
            } else if (config.verboseLogs) {
                console.log(chalk.green(`Topic: ${topicName} ⇢ Similarity Score: ${similarity.toFixed(4)}`));
            }
        }
    }
}


// ------------------------------
// -- Prompt user and run test --
// ------------------------------
const [yOption, nOption] = config.verboseLogs ? ['Y', 'n'] : ['y', 'N'];
rl.question(`\nDisplay verbose logs? (${yOption}/${nOption}): `, async (answer) => {
    const verboseAnswer = answer.trim().toLowerCase();
    config.verboseLogs = verboseAnswer === '' ? config.verboseLogs : verboseAnswer.startsWith('y');
    
    // If verbose logging is disabled, ask about showing matches
    if (!config.verboseLogs) {
        const [matchYOption, matchNOption] = config.showMatches ? ['Y', 'n'] : ['y', 'N'];
        await new Promise(resolve => {
            rl.question(`\nDisplay matched sentences? (${matchYOption}/${matchNOption}): `, (matchAnswer) => {
                const matchVerboseAnswer = matchAnswer.trim().toLowerCase();
                config.showMatches = matchVerboseAnswer === '' ? config.showMatches : matchVerboseAnswer.startsWith('y');
                fs.writeFileSync(configPath, JSON.stringify(config, null, 4));
                resolve();
            });
        });
    }

    fs.writeFileSync(configPath, JSON.stringify(config, null, 4));

    // Now show available test messages
    console.log('\nAvailable test messages:');
    testMessageFiles.forEach((file, index) => {
        const marker = (index + 1) === config.lastTestMessage ? ' ← (last used)' : '';
        console.log(`${index + 1}. ${file}${marker}`);
    });

    rl.question(`\nEnter the number of the test message to analyze (press Enter for ${config.lastTestMessage}): `, async (answer) => {
        const fileIndex = answer.trim() === '' 
            ? config.lastTestMessage - 1 
            : parseInt(answer) - 1;
        
        if (fileIndex >= 0 && fileIndex < testMessageFiles.length) {
            // Save selection to config
            config.lastTestMessage = fileIndex + 1;
            fs.writeFileSync(configPath, JSON.stringify(config, null, 4));

            const testMessage = fs.readFileSync(`test-messages/${testMessageFiles[fileIndex]}`, 'utf8');
            const startTime = performance.now();
            await testSimilarity(testMessage);
            const endTime = performance.now();

            // summary logs
            console.log(chalk.blue('\n-----------------------------------------------------\n'));
            console.log(`Time taken: ${((endTime - startTime) / 1000).toFixed(3)} seconds`);
            console.log(`Total sentences: ${totalSentences}`);
            console.log(`Total comparisons: ${totalComparisons}`);
            console.log(`Number of Matches: ${sentenceMatches.length}`);
            console.log(`Embedding Model: ${topicEmbeddings[0].topicEmbedding.embeddingModel}`);
            console.log(`Embedding Precision: ${topicEmbeddings[0].topicEmbedding.modelPrecision}`);
        } else {
            console.log('Invalid selection');
        }
        
        rl.close();
    });
});