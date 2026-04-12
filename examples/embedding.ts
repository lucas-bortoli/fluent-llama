import { Client } from "../dist/client.js";
import { Logger } from "./logger.ts";

const logger = new Logger("EmbeddingExample");

async function main() {
  try {
    const client = await Client.from("http://127.0.0.1:9001");

    // Load the embedding model
    await client.load("MiniLM-L6-v2");
    logger.info("Model loaded successfully!");

    // Create embedding model
    const embeddingModel = await client.createEmbeddingModel("MiniLM-L6-v2");

    // Test 1: Single text embedding
    logger.info("\n=== Test 1: Single text embedding ===");
    const result1 = await embeddingModel.embed({
      input: "Hello, world!",
      normalization: "L2",
    });

    logger.info(`Generated ${result1.length} dimensions`);
    logger.info(`Embedding dimension: ${result1.length}`);

    // Test 2: Multiple texts
    logger.info("\n=== Test 2: Multiple texts ===");
    const result2 = await embeddingModel.embed({
      input: ["The quick brown fox", "A fast brown fox jumps"],
      normalization: "L2",
    });

    logger.info(`Generated ${result2.length} embedding(s)`);
    logger.info(`Embedding dimension: ${result2[0]!.length}`);

    // Test 3: No normalization
    logger.info("\n=== Test 3: No normalization ===");
    const result3 = await embeddingModel.embed({
      input: "Unnormalized embedding test",
      normalization: null,
    });

    logger.info(`Generated ${result3.length} dimensions`);
    logger.info(`Embedding dimension: ${result3.length}`);

    // Test 4: L1 normalization
    logger.info("\n=== Test 4: L1 normalization ===");
    const result4 = await embeddingModel.embed({
      input: "L1 normalized text",
      normalization: "L1",
    });

    logger.info(`Generated ${result4.length} dimensions`);
    logger.info(`Embedding dimension: ${result4.length}`);

    // Test 5: MaxAbs normalization
    logger.info("\n=== Test 5: MaxAbs normalization ===");
    const result5 = await embeddingModel.embed({
      input: "MaxAbs normalized text",
      normalization: "MaxAbs",
    });

    logger.info(`Generated ${result5.length} dimensions`);
    logger.info(`Embedding dimension: ${result5.length}`);

    logger.info("\n✅ All tests passed!");

    // Cleanup: unload the model
    await client.unload("MiniLM-L6-v2");
    logger.info("Model unloaded.");
  } catch (error) {
    logger.error(error);
  }
}

main();
