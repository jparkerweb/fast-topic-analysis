// -------------
// -- imports --
// -------------
import { prefixConfig, generateEmbeddings } from "./modules/embedding.js";
import { cosineSimilarity } from "./modules/similarity.js";
import { parseSentences } from 'sentence-parse';
import fs from 'fs';
import readline from 'readline';
import chalk from 'chalk';
import path from 'path';

// Create readline interface
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

// Parse command line arguments
const args = parseCommandLineArgs();

// Load or create config
let config = { lastTestMessage: 1, verboseLogs: true, showMatches: true };
const configPath = './run-demo-config.json';
try {
    config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
} catch {
    fs.writeFileSync(configPath, JSON.stringify(config, null, 4));
}

// Apply command line arguments to config
if (args.verbose !== undefined) config.verboseLogs = true;
if (args.quiet !== undefined) config.verboseLogs = false;
if (args.showMatches !== undefined) config.showMatches = true;
if (args.hideMatches !== undefined) config.showMatches = false;

// Load topic embeddings from `data/topic_embeddings` folder
const topicEmbeddingsDir = 'data/topic_embeddings';
const topicEmbeddingFiles = fs.readdirSync(topicEmbeddingsDir).filter(file => file.endsWith('.json'));

// Group embeddings by topic
const topicEmbeddings = {};
topicEmbeddingFiles.forEach(file => {
    const filePath = path.join(topicEmbeddingsDir, file);
    const embeddingData = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    const topicName = embeddingData.topic;
    
    if (!topicEmbeddings[topicName]) {
        topicEmbeddings[topicName] = [];
    }
    
    topicEmbeddings[topicName].push({
        clusterIndex: embeddingData.clusterIndex,
        totalClusters: embeddingData.totalClusters,
        clusterSize: embeddingData.clusterSize,
        clusterCoverage: embeddingData.clusterCoverage,
        cohesion: embeddingData.cohesion || "N/A",
        threshold: embeddingData.threshold,
        embedding: embeddingData.embedding
    });
});

// Log topic embedding information
console.log(chalk.blue('\nLoaded topic embeddings:'));
Object.keys(topicEmbeddings).forEach(topicName => {
    console.log(chalk.green(`  - ${topicName}: ${topicEmbeddings[topicName].length} clusters`));
});

// Load test message files
const testMessageFiles = fs.readdirSync('test-messages')
    .sort((a, b) => {
        const numA = parseInt(a.match(/\d+/)[0]);
        const numB = parseInt(b.match(/\d+/)[0]);
        return numA - numB;
    });

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
    let sentencesWithEmbeddings = await generateEmbeddings(sentences, {
        prefix: prefixConfig.queryPrefix,
        returnPhrases: true,
        logging: false,
    });

    for (const { phrase, embedding } of sentencesWithEmbeddings) {
        let matchFound = false;
        let bestMatch = { topicName: null, similarity: 0, clusterIndex: null };
        
        if (config.verboseLogs) {
            console.log(`\nSentence: ${phrase}`);
        }
        
        // Compare with all topic embeddings
        for (const topicName in topicEmbeddings) {
            const topicClusters = topicEmbeddings[topicName];
            const threshold = topicClusters[0].threshold; // All clusters for a topic have the same threshold
            
            // Compare with each cluster for this topic
            for (const cluster of topicClusters) {
                totalComparisons++;
                
                const similarity = cosineSimilarity(embedding, cluster.embedding);
                
                // Track best match across all topics and clusters
                if (similarity > bestMatch.similarity) {
                    bestMatch = { 
                        topicName, 
                        similarity, 
                        clusterIndex: cluster.clusterIndex,
                        totalClusters: cluster.totalClusters,
                        cohesion: cluster.cohesion,
                        threshold
                    };
                }
                
                if (similarity >= threshold) {
                    matchFound = true;
                    
                    // Clean phrase by removing prefixConfig.queryPrefix from phrase
                    const cleanedPhrase = (phrase.startsWith(prefixConfig.queryPrefix) && prefixConfig.queryPrefix !== '')
                        ? phrase.slice(prefixConfig.queryPrefix.length)
                        : phrase;
                    
                    sentenceMatches.push({ 
                        topicName, 
                        cleanedPhrase,
                        clusterIndex: cluster.clusterIndex,
                        totalClusters: cluster.totalClusters,
                        cohesion: cluster.cohesion
                    });
                    
                    if (config.verboseLogs) {
                        console.log(chalk.red(`Topic: ${topicName} (Cluster ${cluster.clusterIndex + 1}/${cluster.totalClusters}, Cohesion: ${cluster.cohesion}) ⇢ Similarity Score: ${similarity.toFixed(4)}`));
                    } else if(!config.verboseLogs && config.showMatches) {
                        console.log(chalk.red(`Topic: ${topicName} (Cluster ${cluster.clusterIndex + 1}/${cluster.totalClusters}, Cohesion: ${cluster.cohesion}) ⇢ Similarity Score: ${similarity.toFixed(4)} ⇠ ${cleanedPhrase}`));
                    }
                    
                    // We found a match for this topic, no need to check other clusters
                    break;
                } else if (config.verboseLogs) {
                    console.log(chalk.green(`Topic: ${topicName} (Cluster ${cluster.clusterIndex + 1}/${cluster.totalClusters}, Cohesion: ${cluster.cohesion}) ⇢ Similarity Score: ${similarity.toFixed(4)}`));
                }
            }
        }
        
        // If no match was found but we want to show the best match anyway
        if (!matchFound && config.verboseLogs && bestMatch.topicName) {
            console.log(chalk.yellow(`Best match (below threshold): ${bestMatch.topicName} (Cluster ${bestMatch.clusterIndex + 1}/${bestMatch.totalClusters}, Cohesion: ${bestMatch.cohesion}) ⇢ Score: ${bestMatch.similarity.toFixed(4)} (Threshold: ${bestMatch.threshold})`));
        }
    }
}


// ------------------------------
// -- Prompt user and run test --
// ------------------------------
const [yOption, nOption] = config.verboseLogs ? ['Y', 'n'] : ['y', 'N'];
rl.question(`\nDisplay verbose logs? (${yOption}/${nOption}): `, async (answer) => {
    const verboseAnswer = answer.trim().toLowerCase();
    
    if (verboseAnswer === 'y') {
        config.verboseLogs = true;
    } else if (verboseAnswer === 'n') {
        config.verboseLogs = false;
    }
    
    const [showOption, hideOption] = config.showMatches ? ['Y', 'n'] : ['y', 'N'];
    rl.question(`Show matched sentences? (${showOption}/${hideOption}): `, async (answer) => {
        const showAnswer = answer.trim().toLowerCase();
        
        if (showAnswer === 'y') {
            config.showMatches = true;
        } else if (showAnswer === 'n') {
            config.showMatches = false;
        }
        
        // Save config
        fs.writeFileSync(configPath, JSON.stringify(config, null, 4));
        
        // Determine which test message to use
        let testMessagePath;
        if (args.file) {
            if (isNaN(args.file)) {
                // If it's not a number, treat it as a filename
                testMessagePath = args.file;
                if (!fs.existsSync(testMessagePath)) {
                    testMessagePath = `test-messages/${args.file}`;
                }
            } else {
                // If it's a number, treat it as an index
                const messageIndex = parseInt(args.file);
                if (messageIndex > 0 && messageIndex <= testMessageFiles.length) {
                    testMessagePath = `test-messages/${testMessageFiles[messageIndex - 1]}`;
                    config.lastTestMessage = messageIndex;
                } else {
                    console.log(chalk.red(`Invalid test message number. Please choose between 1 and ${testMessageFiles.length}.`));
                    rl.close();
                    return;
                }
            }
        } else {
            rl.question(`Which test message? (1-${testMessageFiles.length}) [${config.lastTestMessage}]: `, async (answer) => {
                let messageIndex = config.lastTestMessage;
                
                if (answer.trim() !== '') {
                    messageIndex = parseInt(answer.trim());
                    if (isNaN(messageIndex) || messageIndex < 1 || messageIndex > testMessageFiles.length) {
                        console.log(chalk.red(`Invalid test message number. Using default: ${config.lastTestMessage}`));
                        messageIndex = config.lastTestMessage;
                    }
                }
                
                config.lastTestMessage = messageIndex;
                fs.writeFileSync(configPath, JSON.stringify(config, null, 4));
                
                const testMessagePath = `test-messages/${testMessageFiles[messageIndex - 1]}`;
                const testMessage = fs.readFileSync(testMessagePath, 'utf8');
                
                console.log(chalk.blue(`\nAnalyzing test message ${messageIndex}: ${testMessageFiles[messageIndex - 1]}`));
                
                const startTime = Date.now();
                await testSimilarity(testMessage);
                const endTime = Date.now();
                
                // Print summary
                console.log(chalk.blue('\n-----------------------------------------------------\n'));
                console.log(chalk.yellow(`Analysis completed in ${(endTime - startTime) / 1000} seconds`));
                console.log(chalk.yellow(`Total sentences analyzed: ${totalSentences}`));
                console.log(chalk.yellow(`Total comparisons performed: ${totalComparisons}`));
                console.log(chalk.yellow(`Total matches found: ${sentenceMatches.length}`));
                
                if (sentenceMatches.length > 0 && config.showMatches) {
                    console.log(chalk.blue('\nMatched Sentences:'));
                    sentenceMatches.forEach(match => {
                        console.log(chalk.green(`  - ${match.topicName} (Cluster ${match.clusterIndex + 1}/${match.totalClusters}, Cohesion: ${match.cohesion}): ${match.cleanedPhrase}`));
                    });
                }
                
                rl.close();
            });
            return;
        }
        
        // If we have a direct file path, process it
        if (testMessagePath) {
            if (!fs.existsSync(testMessagePath)) {
                console.log(chalk.red(`Test message file not found: ${testMessagePath}`));
                rl.close();
                return;
            }
            
            const testMessage = fs.readFileSync(testMessagePath, 'utf8');
            console.log(chalk.blue(`\nAnalyzing test message: ${path.basename(testMessagePath)}`));
            
            const startTime = Date.now();
            await testSimilarity(testMessage);
            const endTime = Date.now();
            
            // Print summary
            console.log(chalk.blue('\n-----------------------------------------------------\n'));
            console.log(chalk.yellow(`Analysis completed in ${(endTime - startTime) / 1000} seconds`));
            console.log(chalk.yellow(`Total sentences analyzed: ${totalSentences}`));
            console.log(chalk.yellow(`Total comparisons performed: ${totalComparisons}`));
            console.log(chalk.yellow(`Total matches found: ${sentenceMatches.length}`));
            
            if (sentenceMatches.length > 0 && config.showMatches) {
                console.log(chalk.blue('\nMatched Sentences:'));
                sentenceMatches.forEach(match => {
                    console.log(chalk.green(`  - ${match.topicName} (Cluster ${match.clusterIndex + 1}/${match.totalClusters}, Cohesion: ${match.cohesion}): ${match.cleanedPhrase}`));
                });
            }
            
            rl.close();
        }
    });
});

// ----------------------------------
// -- Parse command line arguments --
// ----------------------------------
function parseCommandLineArgs() {
    const args = {};
    const argv = process.argv.slice(2);
    
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        
        if (arg === '--verbose' || arg === '-v') {
            args.verbose = true;
        } else if (arg === '--quiet' || arg === '-q') {
            args.quiet = true;
        } else if (arg === '--show-matches' || arg === '-s') {
            args.showMatches = true;
        } else if (arg === '--hide-matches' || arg === '-h') {
            args.hideMatches = true;
        } else if (arg === '--help') {
            printHelp();
            process.exit(0);
        } else if (!arg.startsWith('-')) {
            args.file = arg;
        }
    }
    
    return args;
}

// -----------------------
// -- Print help message --
// -----------------------
function printHelp() {
    console.log(`
Usage: node run-demo.js [options] [file]

Options:
  --verbose, -v       Enable verbose logging
  --quiet, -q         Disable verbose logging
  --show-matches, -s  Show matched sentences
  --hide-matches, -h  Hide matched sentences
  --help              Show this help message

Arguments:
  file                Test message file to analyze (number or filename)

Examples:
  node run-demo.js 2
  node run-demo.js message-1.txt
  node run-demo.js --quiet --show-matches
`);
}