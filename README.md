# fluent-llama

[![npm](https://img.shields.io/npm/v/@lucas-bortoli/fluent-llama)](https://www.npmjs.com/package/@lucas-bortoli/fluent-llama)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue)](https://www.typescriptlang.org/)

> This package is currently in **Alpha status**. It is not yet suitable for production use. Breaking changes may occur without notice.

**fluent-llama** is a type-safe, fluent API client for interacting with `llama-server` (llama.cpp inference server). It provides a modern, expressive interface for chat completions, tool calling, vision tasks, and agent loops.

## Features

- **Fluent Configuration** 🧠: Builder pattern for `Sampling` and `Toolset` configurations.
- **Agent Loops** 🤖: The `act()` method handles the multi-turn reasoning and tool execution cycle automatically.
- **Error Handling with supermacro's neverthrow** 🛡️: All operations return `Result<T, E>` types from the [`neverthrow`](https://github.com/supermacro/neverthrow) library. Handle errors explicitly with TypeScript support. Highly recommend checking it out.
- **Vision Support** 📷: Native handling of image attachments via Base64.
- **Reasoning** 🔍: Supports `reasoningContent` (Chain of Thought) streams.
- **Advanced Sampling** ⚙️: Fine-grained control over temperature, top-k, top-p, mirostat, DRY, XTC, and more.
- **Streaming** 🔄: Full SSE (Server-Sent Events) support for real-time token streaming.

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
  // Connect to your local llama-server
  const client = await Client.from("http://localhost:8080");
  const llmResult = await client.createTextModel("Qwen3.5-35B-A3B");

  if (llmResult.isErr()) {
    console.error("Error opening client:", llmResult.error);
    process.exit(1);
  }

  const llm = llmResult.value;

  const result = await llm.respond({
    instructions: "You are a helpful assistant.",
    history: [{ role: "user", content: "Hello, who are you?", attachments: [] }],
    sampling: new Sampling().setSeed(RandomSeed).build(),
  });

  if (result.isOk()) {
    console.log(result.value.response.content);
  } else {
    // Always handle errors explicitly with neverthrow patterns
    console.error(result.error);
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

const client = await Client.from("http://localhost:8080");
const llmResult = await client.createTextModel("Qwen3.5-35B-A3B");

if (llmResult.isErr()) {
  console.error("Error opening client:", llmResult.error);
  process.exit(1);
}

const llm = llmResult.value;

// Run the agent loop
const response = await llm.act({
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

if (response.isOk()) {
  // The history contains the new generated messages with tool results
  console.log(response.value);
} else {
  // Handle errors explicitly
  console.error(response.error);
}
```

### 3. Vision Support

You can send images by attaching binary content to user messages.

```typescript
import fs from "node:fs/promises";
import path from "node:path";

// ...obtain a TextModel instance like before...

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
```

## Error Handling with Neverthrow

This library uses [`neverthrow`](https://github.com/supermacro/neverthrow) for all error handling. **Every fallible operation returns a `Result<T, E>`** instead of throwing errors. This means you must handle errors explicitly.

### Understanding `Result<T, E>`

- `Result.isOk()` → Check if the operation succeeded
- `Result.isErr()` → Check if the operation failed
- `Result.value` → Access the successful result (only when `isOk()`)
- `Result.error` → Access the error (only when `isErr()`)

Check neverthrow's documentation for more information.

### Example: Handling Different Error Types

```typescript
const response = await llm.respond({
  instructions: "You are a helpful assistant.",
  history: [
    /* ... */
  ],
  sampling: new Sampling().build(),
});

if (response.isErr()) {
  const error = response.error;

  switch (error.kind) {
    case "EmptyMessageHistory":
      console.error("No conversation history provided");
      break;
    case "RequestAborted":
      console.error("Request was cancelled before completion");
      break;
    case "ServerError":
      console.error("Server returned unexpected error:", error.cause);
      break;
    case "RequestError":
      console.error("API request failed:", {
        status: error.httpStatusCode,
        details: error.details,
      });
      break;
    case "InvalidParameter":
      console.error("Invalid parameters provided:", error.details);
      break;
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

This package is built specifically for the API interface exposed by `llama-server` (llama.cpp). Some endpoints use the OpenAI compat layer, but this package is **specifically geared towards llama-server's API. Do not use this package with other OpenAI-compatible servers**.

## Disclaimer

**This software is in Alpha.**

- Stability is not guaranteed.
- API endpoints or types may change.
- Do not use in production environments until a stable version is released.

## License

MIT License. See [LICENSE](LICENSE) for details.
