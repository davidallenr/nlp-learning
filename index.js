// index.js
const express = require("express");
const cors = require("cors");
const { trainModel, classifyText } = require("./models");
const app = express();
app.use(cors());
app.use(express.json());

// Train the model at the start
(async () => {
  await trainModel();
})();

// Endpoint to classify text
app.post("/classify", async (req, res) => {
  const text = req.body.text;
  if (!text) {
    return res.status(400).send({ error: "Missing text in request body" });
  }
  try {
    const domain = await classifyText(text);
    res.send({ domain: domain });
  } catch (err) {
    res.status(500).send({ error: "Error classifying text" });
  }
});

// Start the server
const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`Server listening on port ${port}`));
