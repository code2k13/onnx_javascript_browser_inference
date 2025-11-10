import express from "express";
import { WebSocketServer } from "ws";
import ort from "onnxruntime-node";
import { AutoTokenizer } from "@huggingface/transformers";

// --- Softmax + Top-K utilities ---
function softmax(logits, temperature = 1.0) {
  const adjusted = logits.map(l => l / temperature);
  const max = Math.max(...adjusted);
  const exp = adjusted.map(l => Math.exp(l - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(e => e / sum);
}

function topKSample(probabilities, k) {
  const indexed = probabilities.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p);
  const top = indexed.slice(0, k);
  const sum = top.reduce((a, b) => a + b.p, 0);
  const norm = top.map(t => t.p / sum);
  let r = Math.random(), cum = 0;
  for (let i = 0; i < norm.length; i++) {
    cum += norm[i];
    if (r < cum) return top[i].i;
  }
  return top[0].i;
}

// --- Load model + tokenizer ---
console.log("Loading model and tokenizer...");
const tokenizer = await AutoTokenizer.from_pretrained("./model_files");
const session = await ort.InferenceSession.create("./model_files/model.onnx");
const outputEndId = (await tokenizer.encode("<|OUTPUT_END|>"))[0];
console.log("✅ Model and tokenizer loaded");

// --- Express + WebSocket setup ---
const app = express();
app.use(express.static(".")); // serves index.html from same folder
const server = app.listen(3000, () =>
  console.log("➡️  Open http://localhost:3000")
);
const wss = new WebSocketServer({ server });

wss.on("connection", ws => {
  console.log("🟢 New WebSocket connection");

  ws.on("message", async msg => {
    const prompt = msg.toString().trim();
    console.log("Received prompt:", prompt);
    try {
      await streamParaphrase(ws, prompt);
      // ✅ Flush context buffer — reset after done
      ws.send("\n\n✅ Done");
      
    } catch (err) {
      console.error("❌ Error:", err);
      ws.send("❌ Error during generation");
    }
  });
});

async function streamParaphrase(ws, prompt) {
  const maxTokens = 50, topK = 8, temperature = 0.8;

  // 🔹 single-shot input text — no previous context carried over
  let inputText = `Rephrase the following:\nInput: ${prompt}\nOutput: `;
  let currentTokens = 0;

  for (let step = 0; step < maxTokens; step++) {
    const encoded = await tokenizer(inputText, { return_tensors: "ort" });
    const feeds = { input_ids: encoded.input_ids.ort_tensor };
    if (encoded.attention_mask)
      feeds.attention_mask = encoded.attention_mask.ort_tensor;

    const outputs = await session.run(feeds);
    const logits = outputs.logits?.data || outputs[Object.keys(outputs)[0]].data;
    const inputData = encoded.input_ids.data || encoded.input_ids.ort_tensor.data;
    const vocabSize = logits.length / inputData.length;

    const lastLogits = Array.from(logits.slice(-vocabSize), Number);
    const probs = softmax(lastLogits, temperature);
    const nextId = topKSample(probs, topK);

    const nextToken = await tokenizer.decode([nextId], {
      skip_special_tokens: false
    });

    if (
      nextToken &&
      !["<|OUTPUT_START|>", "<|OUTPUT_END|>", "Output:"].includes(nextToken.trim())
    ) {
      ws.send(nextToken);
      process.stdout.write(nextToken);
    }

    inputText += nextToken;
    currentTokens++;

    if (nextId === outputEndId || currentTokens >= maxTokens) break;
  }

  // ✅ Context flush handled by caller
  console.log(process.memoryUsage());
}
