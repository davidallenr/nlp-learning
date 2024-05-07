const tf = require("@tensorflow/tfjs-node");
const {
  domains,
  lexicon,
  encodeInput,
  prepareData,
} = require("./modelManager");
const { tokenize } = require("../utils/tokenizer");

let loadedModel;

function createModel(lexiconLength, numDomains) {
  const model = tf.sequential();
  const initialLearningRate = 0.01;

  model.add(
    tf.layers.dense({
      units: 128,
      activation: "relu",
      inputShape: [lexiconLength],
    })
  );
  model.add(tf.layers.dropout({ rate: 0.5 })); // Add dropout layer
  model.add(
    tf.layers.dense({
      units: 64,
      activation: "relu",
    })
  );
  model.add(tf.layers.dropout({ rate: 0.5 })); // Add dropout layer
  model.add(
    tf.layers.dense({
      units: numDomains,
      activation: "softmax",
    })
  );
  const optimizer = tf.train.adam(initialLearningRate);
  model.compile({
    optimizer: optimizer,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

async function trainModel() {
  const { xTrain, yTrain, xVal, yVal } = prepareData(); // Retrieve validation data
  const model = createModel(lexicon.length, domains.length);

  await model.fit(xTrain, yTrain, {
    epochs: 300,
    batchSize: 16,
    validationData: [xVal, yVal],
    callbacks: [
      tf.callbacks.earlyStopping({ monitor: "val_loss", patience: 20 }),
    ],
  });

  console.log("Model trained successfully!");
  await model.save("file://./model-save");
}

async function loadModel() {
  if (!loadedModel) {
    try {
      loadedModel = await tf.loadLayersModel("file://./model-save/model.json");
    } catch (err) {
      console.error("Error loading model:", err);
    }
  }
  return loadedModel;
}

async function classifyText(text) {
  const model = await loadModel();
  const processedInput = encodeInput(tokenize(text));
  const predictions = model.predict(tf.tensor2d([processedInput]));

  console.log(`Raw predictions before softmax: ${predictions.dataSync()}`);

  // Safe softmax computation to avoid numerical issues
  const softmaxScores = tf.tidy(() => {
    const predictionsExp = tf.exp(predictions.sub(tf.max(predictions)));
    return predictionsExp.div(predictionsExp.sum());
  });

  const softmaxScoresArray = softmaxScores.dataSync();
  console.log(`Softmax scores: ${softmaxScoresArray}`);

  const predictedIndex = softmaxScores.argMax(1).dataSync()[0];
  const confidenceScore = softmaxScoresArray[predictedIndex] * 100;

  console.log(
    `Predicted domain: ${
      domains[predictedIndex]
    } with confidence: ${confidenceScore.toFixed(2)}%`
  );
  return { domain: domains[predictedIndex], confidence: confidenceScore };
}

module.exports = { trainModel, classifyText };
