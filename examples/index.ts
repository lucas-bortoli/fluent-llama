import fs from "node:fs/promises";
import path from "node:path";
import * as v from "valibot";
import { Client } from "../dist/client.js";
import { RandomSeed, Sampling } from "../dist/sampling.js";
import { tool, Toolset } from "../dist/tool.js";
import Logger from "./logger.ts";

const logger = new Logger("Example");

/**
 * A tool to query weather data for a specific location.
 */
export const weather = tool({
  name: "query-weather",
  description: "Gets weather data for a given location.",
  parameters: { location: v.string() },
  exec: async ({ location }) => {
    logger.info("Weather tool.", { location });
    return { location, current: "Rain", temperature: location.length, units: "Celsius" };
  },
});

/**
 * A tool to perform web searches via DuckDuckGo.
 */
export const webSearch = tool({
  name: "web-search",
  description:
    "Queries DuckDuckGo for a given search query. Returns a list of relevant page titles and URLs.",
  parameters: { query: v.string() },
  exec: async ({ query }) => {
    logger.info("Web search tool.", { query });
    return [
      { title: "AD: Click Here!", url: "http://ad.iru1a.uk/target/4312737698" },
      {
        title: "Bethesda drops the Creation Engine",
        url: "http://publick.dc/news/gaming-bethesda-drops-creation-engine",
      },
    ];
  },
});

/**
 * A tool to execute sandboxed JavaScript code.
 */
export const jsEval = tool({
  name: "js",
  description:
    "Safely runs sandboxed JavaScript functions. If you need access to the script's output, return it as the function's return value.",
  parameters: { code: v.string() },
  exec: async ({ code }) => {
    logger.info("JavaScript execution.", { code });
    return {
      result: eval(code), // evil
    };
  },
});

/**
 * Example usage script for the LLM Client and TextModel.
 *
 * Demonstrates initialization, image attachment, and agent execution with tools.
 */
const imageData = await fs.readFile(path.join(import.meta.dirname, "./skyrim.jpg"));

const clientResult = await Client.from("http://localhost:9001");

if (clientResult.isErr()) {
  logger.error("Failed to create client:", clientResult.error);
  process.exit(1);
}

const client = clientResult.value;

logger.info("Client info:", client);

const llmResult = await client.createTextModel("Qwen3.5-35B-A3B");

if (llmResult.isErr()) {
  logger.error("Failed to create model:", llmResult.error);
  process.exit(1);
}

const llm = llmResult.value;

const response = await llm.act({
  instructions: "You are a helpful assistant. Use the tools given to help the user.",
  history: [
    {
      role: "user",
      content:
        "What do you see? Once identified, please search the web for the image's description",
      attachments: [{ mimeType: "image/jpg", content: imageData.buffer }],
    },
  ],
  sampling: new Sampling()
    .setSeed(RandomSeed)
    //.setTokenBias(248069, 11.8)
    //.setGrammar({
    //  type: "Gbnf",
    //  grammar: "root ::= pre <[248069]> post\npre ::= !<[248069]>*\npost ::= !<[248069]>*",
    //})
    .setSamplerPresencePenalty(1.1)
    .build(),
  toolset: new Toolset([weather, webSearch, jsEval])
    .setInvocationRequirement({ mode: "RequireOneSpecific", tool: "query-weather" })
    .setWhitelist(["query-weather"])
    .setBatchMode("Parallel")
    .setCallbackOnToolCallStart((tool) => {})
    .setCallbackOnToolCallEnd((tool) => {})
    .build(),
  onEvent: (data) => {
    if (data.kind === "ToolCallToken") {
      if ("callIdFragment" in data) process.stderr.write(`\n${data.callIdFragment}: `);
      if ("functionNameFragment" in data) process.stderr.write(`${data.functionNameFragment}\n`);
      if ("argumentFragment" in data) process.stderr.write(data.argumentFragment);
    } else {
      process.stderr.write(data.fragment);
    }
  },
});

if (response.isOk()) {
  logger.debug("Response:", response.value);
} else {
  logger.error("Error", response.error);
}
