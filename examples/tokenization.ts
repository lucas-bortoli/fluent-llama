import { Client } from "../dist/client.js";
import Logger from "./logger.ts";

const logger = new Logger("TokenizationExample");

async function main() {
  try {
    const client = await Client.from("http://127.0.0.1:9001");

    // load a text model for tokenization
    await client.load("Qwen3.6-35B-A3B");
    const textModel = await client.createTextModel("Qwen3.6-35B-A3B");
    logger.info("Model created successfully!");

    logger.info("Basic tokenization");
    const tokens = await textModel.tokenize({
      text: "Hello, world!",
    });

    logger.info(`Tokenized "${"Hello, world!"}" into ${tokens.length} tokens`);
    logger.info(`Token IDs: ${tokens.join(", ")}`);

    // tokenization with special tokens
    logger.info("Tokenization with special tokens");
    const tokensWithSpecial = await textModel.tokenize({
      text: "Hello, world!",
      addSpecial: true,
    });

    logger.info(`With special tokens: ${tokensWithSpecial.length} tokens`);
    logger.info(`Token IDs: ${tokensWithSpecial.join(", ")}`);

    // detokenization
    logger.info("Detokenization");
    const detokenized = await textModel.detokenize(tokens);
    logger.info(`Detokenized back: "${detokenized}"`);
    logger.info(`Match: ${detokenized === "Hello, world!"}`);

    // tokenization with pieces
    logger.info("Tokenization with pieces");
    const tokensWithPieces = await textModel.tokenize({
      text: "Hello, world!",
      withPieces: true,
    });

    logger.info(`Token pieces:`, tokensWithPieces);

    // multi-word tokenization
    logger.info("Longer text tokenization");
    const longText = "The quick brown fox jumps over the lazy dog.";
    const longTokens = await textModel.tokenize({ text: longText });
    logger.info(`Tokenized "${longText}" into ${longTokens.length} tokens`);

    // round-trip verification
    logger.info("Round-trip verification");
    const roundTrip = await textModel.detokenize(longTokens);
    logger.info(`Original:  "${longText}"`);
    logger.info(`Round-trip:"${roundTrip}"`);
    logger.info(`Match: ${longText === roundTrip}`);
  } catch (error) {
    logger.error(error);
  }
}

main();
