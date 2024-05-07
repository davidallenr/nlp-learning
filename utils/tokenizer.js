const natural = require("natural");
const stopWords = [
  "a",
  "the",
  "and",
  "of",
  "in",
  "to",
  "is",
  "you",
  "that",
  "it",
  "he",
  "was",
  "for",
  "on",
  "are",
  "as",
  "with",
  "his",
  "they",
  "at",
];

const tokenizer = new natural.WordTokenizer();

const tokenize = (text) => {
  return tokenizer
    .tokenize(text)
    .map((token) => token.toLowerCase()) // Convert to lower case
    .filter((token) => !stopWords.includes(token)); // Remove stop words
};

module.exports = {
  tokenize,
};
