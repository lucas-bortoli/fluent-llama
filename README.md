# fluent-llama

[![npm](https://img.shields.io/npm/v/@lucas-bortoli/fluent-llama)](https://www.npmjs.com/package/@lucas-bortoli/fluent-llama)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue)](https://www.npmjs.com/)

> This package is currently in **Alpha status**. It is not yet suitable for production use. Breaking changes may occur without notice.

**fluent-llama** is a type-safe, fluent API client for interacting with `llama-server` (llama.cpp inference server). It provides a modern, expressive interface for chat completions, tool calling, vision tasks, and agent loops.

## Features

- **Fluent Configuration** 🧠: Builder pattern for `Sampling` and `Toolset` configurations.
- **Agent Loops** 🤖: The `act()` method handles the multi-turn reasoning and tool execution cycle automatically.
- **Error Handling** 🛡️: All operations throw native JavaScript `Error` instances. Handle errors explicitly with standard `try/catch` patterns.
- **Vision Support** 📷: Native handling of image attachments via Base64.
- **Reasoning** 🔍: Supports `reasoningContent` (Chain of Thought) streams.
- **Text Infilling** 📃: Native support for fill-in-the-middle text completion tasks.
- **Advanced Sampling** ⚙️: Fine-grained control over temperature, top-k, top-p, mirostat, DRY, XTC, and more.
- **Streaming** 🔄: Full SSE (Server-Sent Events) support for real-time token streaming.
- **Router Mode** 🛰️: Dynamic model loading/unloading with automatic model discovery.
- **Embeddings** 📊: Generate text embeddings for semantic search, clustering, and similarity tasks.

## Prerequisites

- **Node.js** (v20+ recommended)
- **llama-server**: This client is designed to connect to the OpenAI-compatible API exposed by [`llama-server`](https://github.com/ggerganov/llama.cpp). Ensure your server is running at a compatible version.

## Installation

```bash
npm install @lucas-bortoli/fluent-llama
```

## Quick Start

### 1. Basic Chat Completion

```typescript
import { Client } from "@lucas-bortoli/fluent-llama";
import { RandomSeed, Sampling } from "@lucas-bortoli/fluent-llama";

async function main() {
  try {
    // Connect to your local llama-server
    const client = await Client.from("http://localhost:8080");
    console.log("Client initialized with models:", [...client.modelStatuses.keys()]);

    const llm = await client.createTextModel("Qwen3.5-35B-A3B");

    const result = await llm.respond({
      instructions: "You are a helpful assistant.",
      history: [{ role: "user", content: "Hello, who are you?", attachments: [] }],
      sampling: new Sampling().setSeed(RandomSeed).build(),
    });

    console.log(result.response.content);
  } catch (error) {
    console.error("Error:", error);
  }
}

main();
```

### 2. Tool Calling (Agent Mode)

Use the `act()` method to run autonomous agent loops where the model decides when to use tools.

```typescript
import { Client, tool, Toolset, Sampling, RandomSeed } from "@lucas-bortoli/fluent-llama";
import * as v from "valibot";

// Define a tool using Valibot for schema validation
const weatherTool = tool({
  name: "get_weather",
  description: "Gets weather data for a location.",
  parameters: { location: v.string() },
  exec: async ({ location }) => {
    return { temp: 20, condition: "Sunny" };
  },
});

try {
  const client = await Client.from("http://localhost:8080");
  const llm = await client.createTextModel("Qwen3.5-35B-A3B");

  // Run the agent loop
  const history = await llm.act({
    instructions: "You are a helpful assistant. Use tools to answer.",
    history: [{ role: "user", content: "What's the weather in Tokyo?", attachments: [] }],
    sampling: new Sampling()
      .setSamplerTemperature(0.7)
      .setSamplerTopK(80)
      .setSamplerMinP(0.02)
      .setSeed(RandomSeed)
      .build(),
    toolset: new Toolset([weatherTool]).build(),
  });

  // The history contains the new generated messages with tool results
  console.log(history);
} catch (error) {
  console.error("Agent error:", error);
}
```

### 3. Vision Support

You can send images by attaching binary content to user messages.

```typescript
import fs from "node:fs/promises";
import path from "node:path";
import { Client, Sampling } from "@lucas-bortoli/fluent-llama";

async function main() {
  try {
    const client = await Client.from("http://localhost:8080");
    const llm = await client.createTextModel("Qwen3.5-35B-A3B");

    const imageData = await fs.readFile(path.join(__dirname, "image.jpg"));
    const response = await llm.respond({
      instructions: "Describe this image.",
      history: [
        {
          role: "user",
          content: "What is in this picture?",
          attachments: [{ mimeType: "image/jpeg", content: imageData.buffer }],
        },
      ],
      sampling: new Sampling().build(),
    });

    console.log(response.response.content);
  } catch (error) {
    console.error("Vision error:", error);
  }
}

main();
```

### 4. Model Loading and Unloading (Router Mode)

With llama-server's router mode, you can dynamically load and unload models without restarting the server.

```typescript
import { Client } from "@lucas-bortoli/fluent-llama";

async function main() {
  try {
    const client = await Client.from("http://localhost:8080");

    // Check available models
    console.log("Available models:", [...client.modelStatuses.keys()]);

    // Load a model
    await client.load("Qwen3.5-35B-A3B");
    console.log("Model loaded successfully");

    // Use the model
    const llm = await client.createTextModel("Qwen3.5-35B-A3B");
    const isLoaded = await client.isModelLoaded("Qwen3.5-35B-A3B");
    console.log("Model loaded status:", isLoaded);

    // Unload the model when done
    await client.unload("Qwen3.5-35B-A3B");
    console.log("Model unloaded successfully");
  } catch (error) {
    console.error("Error:", error);
  }
}

main();
```

### 5. Text Infilling

The `predict()` method supports text infilling by using the native `/infill` endpoint. Provide a `prefix`, `suffix`, and the main `prompt` to generate completions for partial text blocks.

```typescript
import { Client, Sampling, RandomSeed } from "@lucas-bortoli/fluent-llama";

async function main() {
  try {
    const client = await Client.from("http://localhost:8080");
    const llm = await client.createTextModel("Qwen3.5-35B-A3B");

    const result = await llm.predict({
      input: {
        prefix: "def sum(a, b):\n",
        suffix: "\n\nprint(sum(5, 8))",
        prompt: "Write this function.",
      },
      sampling: new Sampling().setSamplerTemperature(0.6).setSeed(RandomSeed).build(),
    });

    console.log("Infilling completion:", result.content);
  } catch (error) {
    console.error("Infilling error:", error);
  }
}

main();
```

### 6. Embeddings

Generate text embeddings for semantic search, clustering, and similarity tasks.

```typescript
import { Client, Sampling } from "@lucas-bortoli/fluent-llama";

async function main() {
  try {
    const client = await Client.from("http://localhost:8080");

    // Load an embedding model
    await client.load("all-MiniLM-L6-v2");

    // Create embedding model instance
    const embeddingModel = await client.createEmbeddingModel("all-MiniLM-L6-v2");

    // Generate embedding for single text (returns number[])
    const singleEmbedding = await embeddingModel.embed("Hello, world!");
    console.log("Single embedding dimension:", singleEmbedding.length);
    console.log("First 5 values:", singleEmbedding.slice(0, 5));

    // Generate embeddings for multiple texts (returns number[][])
    const multipleEmbeddings = await embeddingModel.embed([
      "Hello, world!",
      "How are you?",
      "Good morning!",
    ]);

    console.log("Generated", multipleEmbeddings.length, "embeddings");
    console.log("Each embedding has", multipleEmbeddings[0].length, "dimensions");
  } catch (error) {
    console.error("Embedding error:", error);
  }
}

main();
```

## Error Handling

This library uses standard JavaScript `Error` classes for error handling. **Every fallible operation can throw errors**. Handle them explicitly with `try/catch` blocks.

### Available Error Classes

- `ApiRequestError` - API request failures (includes `httpStatusCode` and `responseBody`)
- `InvalidParameterError` - Invalid parameters provided
- `EmptyMessageArrayError` - Empty message history
- `AbortedRequestError` - Request was cancelled
- `UnexpectedServerBehaviorError` - Server returned unexpected response
- `ModelLoadError` - Model load failures
- `ModelUnloadError` - Model unload failures
- `InvalidModelError` - Invalid model ID

### Example: Handling Different Error Types

```typescript
try {
  const response = await llm.respond({
    instructions: "You are a helpful assistant.",
    history: [
      /* ... */
    ],
    sampling: new Sampling().build(),
  });
  console.log(response.response.content);
} catch (error) {
  if (error instanceof InvalidParameterError) {
    console.error("Invalid parameters:", error.message);
  } else if (error instanceof ApiRequestError) {
    console.error("API request failed:", {
      status: error.httpStatusCode,
      responseBody: error.responseBody,
    });
  } else if (error instanceof ModelLoadError) {
    console.error("Model load failed:", error.message, error.inner);
  } else {
    console.error("Unexpected error:", error);
  }
}
```

## Configuration Reference

### Sampling

The `Sampling` class allows you to configure generation parameters fluently.

```typescript
const config = new Sampling()
  .setSamplerTemperature(0.7)
  .setSamplerTopP(0.95)
  .setSamplerTopK(40)
  .setSeed(RandomSeed) // or setSeed(42) for deterministic results
  .setSamplerPresencePenalty(1.0)
  .setGrammar({ type: "Json", schema: { ... } }) // For structured outputs
  .build();
```

### Toolset

The `Toolset` class manages available functions for the LLM.

```typescript
const tools = new Toolset([weatherTool, webSearchTool])
  .setWhitelist(["weather-tool"]) // Only allow these tools
  .setBatchMode("Parallel") // Run tools concurrently
  .setInvocationRequirement("AsNeeded") // Or "RequireOne"
  .build();
```

## Compatibility

This package is built specifically for the API interface exposed by llama-server (llama.cpp). While some endpoints use the OpenAI compat layer, this package **specifically leverages llama-server's native endpoints** for optimal performance and feature support. Do not use this package with other LLM servers.

## Disclaimer

**This software is in Alpha.**

- Stability is not guaranteed.
- API endpoints or types may change.
- Do not use in production environments until a stable version is released.

## License

MIT License. See [LICENSE](LICENSE) for details.
