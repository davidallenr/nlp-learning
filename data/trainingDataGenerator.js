const fs = require("fs");
const trainingData = require("./trainingData.js");

// Function to transform the data into the desired format
function transformData(data) {
  let intents = {};

  // Grouping texts by domain
  data.forEach((item) => {
    if (!intents[item.domain]) {
      intents[item.domain] = [];
    }
    intents[item.domain].push(item.text);
  });

  // Formatting into the specified structure
  let formattedData = {
    intents: [],
  };

  for (const [intent, texts] of Object.entries(intents)) {
    formattedData.intents.push({
      intent: intent,
      text: texts,
    });
  }

  return formattedData;
}

const transformedData = transformData(trainingData);

// Write the output to a JSON file
fs.writeFile(
  "./formattedData.json",
  JSON.stringify(transformedData, null, 2),
  (err) => {
    if (err) {
      console.error("Error writing file:", err);
    } else {
      console.log("File has been written successfully.");
    }
  }
);
