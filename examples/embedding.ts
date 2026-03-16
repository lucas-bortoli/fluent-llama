import { Client } from "../dist/client.js";
import { Logger } from "./logger.ts";

const logger = new Logger("EmbeddingExample");

async function main() {
  const client = await Client.from("http://127.0.0.1:9001");

  if (client.isErr()) {
    logger.error("Failed to connect to llama-server:", client.error);
    if (typeof process !== "undefined") {
      process.exit(1);
    }
    return;
  }

  const clientInstance = client.value;

  // Load the embedding model
  const loadResult = await clientInstance.load("MiniLM-L6-v2");
  if (loadResult.isErr()) {
    logger.error("Failed to load model:", loadResult.error);
    if (typeof process !== "undefined") {
      process.exit(1);
    }
    return;
  }

  logger.info("Model loaded successfully!");

  // Create embedding model
  const embeddingResult = await clientInstance.createEmbeddingModel("MiniLM-L6-v2");
  if (embeddingResult.isErr()) {
    logger.error("Failed to create embedding model:", embeddingResult.error);
    if (typeof process !== "undefined") {
      process.exit(1);
    }
    return;
  }

  const embeddingModel = embeddingResult.value;

  // Test 1: Single text embedding
  logger.info("\n=== Test 1: Single text embedding ===");
  const result1 = await embeddingModel.embed({
    input: "Hello, world!",
    normalization: "L2",
  });

  if (result1.isErr()) {
    logger.error("Failed to generate embedding:", result1.error);
    if (typeof process !== "undefined") {
      process.exit(1);
    }
    return;
  }

  const embedding = result1.value;
  logger.info(`Generated ${embedding.length} dimensions`);
  logger.info(`Embedding dimension: ${embedding.length}`);

  // Test 2: Multiple texts
  logger.info("\n=== Test 2: Multiple texts ===");
  const result2 = await embeddingModel.embed({
    input: ["The quick brown fox", "A fast brown fox jumps"],
    normalization: "L2",
  });

  if (result2.isErr()) {
    logger.error("Failed to generate embeddings:", result2.error);
    if (typeof process !== "undefined") {
      process.exit(1);
    }
    return;
  }

  const embeddings2 = result2.value;
  logger.info(`Generated ${embeddings2.length} embedding(s)`);
  logger.info(`Embedding dimension: ${embeddings2[0]!.length}`);

  // Test 3: No normalization
  logger.info("\n=== Test 3: No normalization ===");
  const result3 = await embeddingModel.embed({
    input: "Unnormalized embedding test",
    normalization: null,
  });

  if (result3.isErr()) {
    logger.error("Failed to generate embedding:", result3.error);
    if (typeof process !== "undefined") {
      process.exit(1);
    }
    return;
  }

  const embedding3 = result3.value;
  logger.info(`Generated ${embedding3.length} dimensions`);
  logger.info(`Embedding dimension: ${embedding3.length}`);

  // Test 4: L1 normalization
  logger.info("\n=== Test 4: L1 normalization ===");
  const result4 = await embeddingModel.embed({
    input: "L1 normalized text",
    normalization: "L1",
  });

  if (result4.isErr()) {
    logger.error("Failed to generate embedding:", result4.error);
    if (typeof process !== "undefined") {
      process.exit(1);
    }
    return;
  }

  const embedding4 = result4.value;
  logger.info(`Generated ${embedding4.length} dimensions`);
  logger.info(`Embedding dimension: ${embedding4.length}`);

  // Test 5: MaxAbs normalization
  logger.info("\n=== Test 5: MaxAbs normalization ===");
  const result5 = await embeddingModel.embed({
    input: "MaxAbs normalized text",
    normalization: "MaxAbs",
  });

  if (result5.isErr()) {
    logger.error("Failed to generate embedding:", result5.error);
    if (typeof process !== "undefined") {
      process.exit(1);
    }
    return;
  }

  const embedding5 = result5.value;
  logger.info(`Generated ${embedding5.length} dimensions`);
  logger.info(`Embedding dimension: ${embedding5.length}`);

  logger.info("\n✅ All tests passed!");

  // Cleanup: unload the model
  await clientInstance.unload("MiniLM-L6-v2");
  logger.info("Model unloaded.");
}

main().catch(logger.error);
