const trainingData = require("../data/trainingData");
const { tokenize } = require("../utils/tokenizer");
const tf = require("@tensorflow/tfjs-node");

// Ensure trainingData is correctly imported and is an array
console.log("Training Data Loaded: ", Array.isArray(trainingData)); // This should log true

// Prepare data
const corpus = trainingData.map((d) => ({
  input: tokenize(d.text),
  output: d.domain,
}));

console.log("Corpus Length: ", corpus.length); // Check if corpus is populated correctly
// Quick check to see what tokenize outputs
const sampleData = "Sample text for tokenization.";
console.log("Tokenization Output: ", tokenize(sampleData));

// Proceed with the rest of your code

const domains = [...new Set(corpus.map((d) => d.output))];
const documents = corpus.map((d) => d.input.join(" ")); // Join tokens to form full documents

// Create a lexicon of all unique words
const lexicon = [...new Set(corpus.flatMap((d) => d.input))];

// Function to calculate TF (Term Frequency)
const calculateTF = (tokens) => {
  const wordCounts = tokens.reduce((acc, word) => {
    acc[word] = (acc[word] || 0) + 1;
    return acc;
  }, {});
  const tokenCount = tokens.length;
  return Object.keys(wordCounts).reduce((acc, word) => {
    acc[word] = wordCounts[word] / tokenCount;
    return acc;
  }, {});
};

// Function to calculate IDF (Inverse Document Frequency)
// Modified IDF calculation to emphasize rarity across domains
const calculateIDF = (documents, minDocFreq = 2) => {
  const docCount = documents.length;
  const wordInDocsCounts = documents.reduce((acc, doc) => {
    new Set(doc.split(" ")).forEach((word) => {
      acc[word] = (acc[word] || 0) + 1;
    });
    return acc;
  }, {});

  return Object.keys(wordInDocsCounts).reduce((acc, word) => {
    if (wordInDocsCounts[word] < minDocFreq) {
      acc[word] = Math.log(docCount / wordInDocsCounts[word]);
    } else {
      acc[word] = 0; // Deprioritize common words
    }
    return acc;
  }, {});
};

const idf = calculateIDF(documents);

// Function to calculate TF-IDF
const calculateTFIDF = (tokens) => {
  const tf = calculateTF(tokens);
  return Object.keys(tf).reduce((acc, word) => {
    acc[word] = tf[word] * (idf[word] || 0);
    return acc;
  }, {});
};

// Function to encode input using TF-IDF
const encodeInput = (tokens) => {
  const tfidfValues = calculateTFIDF(tokens);
  return lexicon.map((word) => tfidfValues[word] || 0);
};

// Prepare the data for training
const prepareData = () => {
  const validationSplit = 0.2;
  // Manually shuffle the corpus using JavaScript
  const shuffledData = corpus
    .map((a) => ({ sort: Math.random(), value: a }))
    .sort((a, b) => a.sort - b.sort)
    .map((a) => a.value);

  console.log("Shuffled Data Length: ", shuffledData.length); // This should log the correct length of the corpus

  const numValidationSamples = Math.floor(
    shuffledData.length * validationSplit
  );
  const numTrainingSamples = shuffledData.length - numValidationSamples;

  const trainingData = shuffledData.slice(0, numTrainingSamples);
  const validationData = shuffledData.slice(numTrainingSamples);

  return {
    xTrain: tf.tensor2d(trainingData.map((d) => encodeInput(d.input))),
    yTrain: tf.tensor2d(
      trainingData.map((d) => {
        const encoded = new Array(domains.length).fill(0);
        encoded[domains.indexOf(d.output)] = 1;
        return encoded;
      })
    ),
    xVal: tf.tensor2d(validationData.map((d) => encodeInput(d.input))),
    yVal: tf.tensor2d(
      validationData.map((d) => {
        const encoded = new Array(domains.length).fill(0);
        encoded[domains.indexOf(d.output)] = 1;
        return encoded;
      })
    ),
  };
};

module.exports = {
  domains,
  lexicon,
  encodeInput,
  prepareData,
};
