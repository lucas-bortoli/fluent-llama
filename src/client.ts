import { toJsonSchema } from "@valibot/to-json-schema";
import * as v from "valibot";
import {
  ApiRequestError,
  requestJson,
  requestStream,
  type ApiChatCompletionOptions,
  type ApiChatCompletionStreamChunk,
  type ApiCompletionStreamChunk,
  type ApiDetokenizeResponse,
  type ApiModelLoadUnloadResponse,
  type ApiModelsResponse,
  type ApiPropsResponse,
  type ApiTimingMetrics,
  type ApiTokenizePiece,
  type ApiTokenizeResponse,
  type FetchFn,
} from "./api.js";
import { EmbeddingModel } from "./embeddingModel.js";
import { objectToCamelCase, objectToSnakeCase } from "./helpers.js";
import {
  prepareHistory,
  type AssistantMessage,
  type History,
  type ToolResultMessage,
} from "./history.js";
import { type SamplingResult } from "./sampling.js";
import { type ToolsetResult } from "./tool.js";

/**
 * Thrown when invalid parameters are passed to the client.
 */
export class InvalidParameterError extends Error {
  constructor(details: string) {
    super(details);
    this.name = "InvalidParameterError";
  }
}

/**
 * Thrown when message history is empty but required.
 */
export class EmptyMessageArrayError extends Error {
  constructor(details: string) {
    super(details);
    this.name = "EmptyMessageArrayError";
  }
}

/**
 * Thrown when a request is aborted by the client.
 */
export class AbortedRequestError extends Error {
  constructor(details: string) {
    super(details);
    this.name = "AbortedRequestError";
  }
}

/**
 * Thrown when the server exhibits unexpected behavior.
 */
export class UnexpectedServerBehaviorError extends Error {
  constructor(details: string, inner?: unknown) {
    super(details);
    this.name = "UnexpectedServerBehaviorError";
    if (inner) {
      this.cause = inner;
    }
  }
}

/**
 * Union type representing all possible errors during text completion (single-turn).
 */
export type CompletionError =
  | EmptyMessageArrayError
  | AbortedRequestError
  | UnexpectedServerBehaviorError;

/**
 * Union type representing all possible errors during chat completion (multi-turn).
 */
export type ChatCompletionError =
  | EmptyMessageArrayError
  | AbortedRequestError
  | UnexpectedServerBehaviorError;

/**
 * Event emitted when a single token fragment is generated during text completion.
 */
export type CompletionCallbackEvent = {
  kind: "Token";
  fragment: string;
};

/**
 * Events emitted during chat completion, including reasoning, content, and tool calls.
 */
export type ChatCompletionCallbackEvent =
  | { kind: "ReasoningToken"; fragment: string; timings?: ApiTimingMetrics }
  | { kind: "ContentToken"; fragment: string; timings?: ApiTimingMetrics }
  | { kind: "ToolCallToken"; callIdFragment: string }
  | { kind: "ToolCallToken"; functionNameFragment: string }
  | { kind: "ToolCallToken"; argumentFragment: string };

/**
 * Helper to execute a tool safely within the model context.
 *
 * Validates arguments using the provided schema and handles execution errors gracefully.
 * @param toolset The available tools registered with the client.
 * @param call The tool call request object parsed from the LLM response.
 * @returns A formatted ToolResultMessage ready for the chat history.
 */
async function executeTool(
  toolset: ToolsetResult,
  call: { id: string; name: string; arguments: string },
): Promise<ToolResultMessage> {
  const tool = toolset.tools.get(call.name);
  if (!tool) {
    return {
      role: "tool",
      toolCallId: call.id,
      content: "Tool not found",
    } satisfies ToolResultMessage;
  }

  try {
    const params = v.parse(
      v.pipe(v.string(), v.parseJson(), tool.parametersSchema),
      call.arguments,
    ) as Record<string, any>;

    const result = await tool.exec(params);

    return {
      role: "tool",
      toolCallId: call.id,
      content: typeof result === "string" ? result : JSON.stringify(result),
    } satisfies ToolResultMessage;
  } catch (error) {
    return {
      role: "tool",
      toolCallId: call.id,
      content: typeof error === "string" ? error : JSON.stringify(error),
    } satisfies ToolResultMessage;
  }
}

/**
 * Result structure returned from the `predict` method.
 */
export interface PredictResult {
  content: string;
  timings: ApiTimingMetrics;
  finishReason: "Word" | "Stop" | "Length";
}

/**
 * Result structure returned from the `respond` method.
 */
export interface RespondResult {
  response: AssistantMessage;
  timings: ApiTimingMetrics;
  finishReason: "Stop" | "ContentFilter" | "Length" | "ToolCallRequest";
}

/**
 * Represents an individual Text Model instance connected to the inference API.
 *
 * Handles model metadata retrieval, text prediction, and chat completion.
 */
export class TextModel {
  public readonly client: Client;
  public readonly id: string;
  public readonly totalSlots: number;
  public readonly contextSize: number;

  private constructor(
    client: Client,
    id: string,
    modelOpts: { totalSlots: number; contextSize: number },
  ) {
    this.client = client;
    this.id = id;
    this.totalSlots = modelOpts.totalSlots;
    this.contextSize = modelOpts.contextSize;
  }

  /**
   * Fetches model metadata from the API and constructs a `TextModel` instance.
   * @param client The Client instance to use for the request.
   * @param id The identifier of the model to load.
   * @returns The `TextModel` instance.
   * @throws ApiRequestError when the API request fails.
   */
  public static async from(client: Client, id: string): Promise<TextModel> {
    const urlWithModel = new URL(client.BASE_URL);
    urlWithModel.searchParams.set("model", id);

    const response = await requestJson<ApiPropsResponse>({
      fetchFn: client.clientOptions.fetchFn,
      baseUrl: urlWithModel,
      method: "GET",
      pathName: "/props",
      transformBody: (body) => objectToCamelCase(body, true),
    });

    const {
      buildInfo,
      totalSlots,
      bosToken,
      eosToken,
      chatTemplateCaps: {
        supportsParallelToolCalls,
        supportsPreserveReasoning,
        supportsStringContent,
        supportsSystemRole,
        supportsToolCalls,
        supportsTools,
        supportsTypedContent,
      },
      defaultGenerationSettings: { nCtx },
    } = response;

    return new TextModel(client, id, { totalSlots, contextSize: nCtx });
  }

  /**
   * Generates a text completion stream for a single prompt.
   *
   * Suitable for tasks that do not require conversation history (e.g., completion, translation).
   * Also supports code infilling when the input is an object with prefix/suffix.
   * @param options Configuration for the prediction including prompt, sampling settings, and callbacks.
   * @returns The completion output and metrics.
   * @throws ApiRequestError, EmptyMessageArrayError, AbortedRequestError, or UnexpectedServerBehaviorError.
   */
  public async predict(options: {
    input:
      | string
      | {
          prefix: string;
          suffix: string;
          extra?: string;
          prompt: string;
        };
    sampling: SamplingResult;
    maxTokens?: number;
    signal?: AbortSignal;
    onEvent?: (data: CompletionCallbackEvent) => void;
  }): Promise<PredictResult> {
    try {
      const body = objectToSnakeCase({
        ...options.sampling,
        ...(typeof options.input === "string"
          ? { prompt: options.input }
          : {
              inputPrefix: options.input.prefix,
              inputSuffix: options.input.suffix,
              inputExtra: options.input.extra,
              prompt: options.input.prompt,
            }),
        maxTokens: options.maxTokens === Infinity ? -1 : (options.maxTokens ?? 64),
        model: this.id,
        stream: true,
      });

      const stream = await requestStream<ApiCompletionStreamChunk>({
        fetchFn: this.client.clientOptions.fetchFn,
        baseUrl: this.client.BASE_URL,
        method: "POST",
        pathName: typeof options.input === "string" ? "/completion" : "/infill",
        body,
        signal: options.signal,
      });

      let content = "";

      for await (const chunk of stream) {
        options.onEvent?.({
          kind: "Token",
          fragment: chunk.content,
        });

        content += chunk.content;

        if (chunk.stop) {
          let finishReason: PredictResult["finishReason"];

          if (chunk.stopType === "word") {
            finishReason = "Word";
          } else if (chunk.stopType === "limit") {
            finishReason = "Length";
          } else if (chunk.stopType === "eos") {
            finishReason = "Stop";
          }

          return {
            content,
            timings: chunk.timings!,
            finishReason: finishReason!,
          };
        }
      }

      throw new UnexpectedServerBehaviorError(
        "The inference server did not provide a completion termination signal",
      );
    } catch (error) {
      if (error instanceof DOMException && error.code === error.ABORT_ERR) {
        throw new AbortedRequestError("The completion request was cancelled before completion");
      } else {
        throw new UnexpectedServerBehaviorError(
          "An unexpected error occurred during inference",
          error,
        );
      }
    }
  }

  /**
   * Generates a chat response based on conversation history and optional tools.
   *
   * Handles tool calls, reasoning content, and multi-turn context.
   * @param options Configuration including instructions, history, sampling, and tools.
   * @returns The response message and finish reason.
   * @throws InvalidParameterError, ApiRequestError, EmptyMessageArrayError, AbortedRequestError, or UnexpectedServerBehaviorError.
   */
  public async respond(options: {
    instructions: string;
    history: History;
    sampling: SamplingResult;
    toolset?: ToolsetResult;
    maxTokens?: number;
    signal?: AbortSignal;
    onEvent?: ((data: ChatCompletionCallbackEvent) => void) | undefined;
  }): Promise<RespondResult> {
    if (options.sampling.grammar !== undefined && options.toolset !== undefined) {
      throw new InvalidParameterError(
        "Grammar-based sampling (JSON or GBNF) and tool calling are mutually exclusive",
      );
    } else if (options.history.length === 0) {
      throw new InvalidParameterError("Conversation history is required for response generation");
    }

    try {
      const body = objectToSnakeCase({
        ...options.sampling,
        messages: prepareHistory(options.instructions, options.history),
        maxTokens: options.maxTokens === Infinity ? -1 : (options.maxTokens ?? 64),
        model: this.id,
        stream: true,
        reasoningFormat: "deepseek",
        tools: options.toolset?.tools
          .entries()
          .map(([name, tool]) => {
            return {
              type: "function",
              function: {
                name,
                description: tool.description,
                parameters: toJsonSchema(tool.parametersSchema),
                strict: true,
              },
            };
          })
          .toArray(),
      }) satisfies ApiChatCompletionOptions;

      const stream = await requestStream<ApiChatCompletionStreamChunk>({
        fetchFn: this.client.clientOptions.fetchFn,
        baseUrl: this.client.BASE_URL,
        method: "POST",
        pathName: "/v1/chat/completions",
        body,
        signal: options.signal,
      });

      const collected = {
        role: "",
        reasoningContent: "",
        content: "",
        toolCalls: [] as { id: string; name: string; arguments: string }[],
      };

      for await (const chunk of stream) {
        const data = chunk.choices[0]!;

        if (data.delta.role) {
          collected.role += data.delta.role;
        }

        if (data.delta.content) {
          collected.content += data.delta.content;
          options.onEvent?.({
            kind: "ContentToken",
            fragment: data.delta.content,
            timings: chunk.timings!,
          });
        }

        if (data.delta.reasoningContent) {
          collected.reasoningContent += data.delta.reasoningContent;
          options.onEvent?.({
            kind: "ReasoningToken",
            fragment: data.delta.reasoningContent,
            timings: chunk.timings!,
          });
        }

        if (data.delta.toolCalls) {
          for (const toolCall of data.delta.toolCalls) {
            if (toolCall.function) {
              if (collected.toolCalls[toolCall.index] === undefined)
                collected.toolCalls[toolCall.index] = { id: "", name: "", arguments: "" };
              const target = collected.toolCalls[toolCall.index]!;
              if (toolCall.id !== undefined) {
                target.id += toolCall.id;
                options.onEvent?.({
                  kind: "ToolCallToken",
                  callIdFragment: toolCall.id,
                });
              }
              if (toolCall.function.name !== undefined) {
                target.name += toolCall.function.name;
                options.onEvent?.({
                  kind: "ToolCallToken",
                  functionNameFragment: toolCall.function.name,
                });
              }
              if (toolCall.function.arguments !== undefined) {
                target.arguments += toolCall.function.arguments;
                options.onEvent?.({
                  kind: "ToolCallToken",
                  argumentFragment: toolCall.function.arguments,
                });
              }
            }
          }
        }

        if (data.finishReason !== null) {
          let finishReason: RespondResult["finishReason"];

          if (data.finishReason === "content_filter") {
            finishReason = "ContentFilter";
          } else if (data.finishReason === "length") {
            finishReason = "Length";
          } else if (data.finishReason === "tool_calls") {
            finishReason = "ToolCallRequest";
          } else {
            finishReason = "Stop";
          }

          const generatedMessage: AssistantMessage = {
            role: "assistant",
            reasoningContent: collected.reasoningContent.length ? collected.reasoningContent : null,
            content: collected.content,
            requestedToolCalls: collected.toolCalls,
          };

          return {
            response: generatedMessage,
            timings: chunk.timings!,
            finishReason: finishReason,
          };
        }
      }

      throw new UnexpectedServerBehaviorError(
        "The inference server did not provide a completion termination signal",
      );
    } catch (error) {
      if (error instanceof DOMException && error.code === error.ABORT_ERR) {
        throw new AbortedRequestError("The completion request was cancelled before completion");
      } else {
        throw new UnexpectedServerBehaviorError(
          "An unexpected error occurred during inference",
          error,
        );
      }
    }
  }

  /**
   * Executes an autonomous agent loop.
   *
   * Repeatedly calls `respond` and executes returned tool calls until no tools are requested.
   * @param options Configuration including instructions, history, tools, and callbacks.
   * @returns The final chat history.
   * @throws ApiRequestError, InvalidParameterError, EmptyMessageArrayError, AbortedRequestError, or UnexpectedServerBehaviorError.
   */
  public async act(options: {
    instructions: string;
    history: History;
    sampling: SamplingResult;
    toolset: ToolsetResult;
    signal?: AbortSignal;
    onEvent?: (data: ChatCompletionCallbackEvent) => void;
  }): Promise<History> {
    const threadHistory: History = [];

    while (true) {
      const response = await this.respond({
        instructions: options.instructions,
        history: [...options.history, ...threadHistory],
        sampling: options.sampling,
        toolset: options.toolset,
        signal: options.signal!,
        onEvent: options.onEvent,
        maxTokens: Infinity,
      });

      threadHistory.push(response.response);

      if (response.finishReason !== "ToolCallRequest") {
        break;
      }

      const toolResults = await Promise.all(
        response.response.requestedToolCalls.map(executeTool.bind(null, options.toolset)),
      );

      threadHistory.push(...toolResults);
    }

    return threadHistory;
  }

  /**
   * Tokenizes text into token IDs with piece metadata.
   * @param options Configuration including text and optional flags.
   * @returns Array of token pieces with IDs and text.
   */
  public async tokenize(options: {
    text: string;
    addSpecial?: boolean;
    parseSpecial?: boolean;
    withPieces: true;
    signal?: AbortSignal;
  }): Promise<ApiTokenizePiece[]>;

  /**
   * Tokenizes text into token IDs.
   * @param options Configuration including text and optional flags.
   * @returns Array of token IDs.
   */
  public async tokenize(options: {
    text: string;
    addSpecial?: boolean;
    parseSpecial?: boolean;
    withPieces?: false;
    signal?: AbortSignal;
  }): Promise<number[]>;

  public async tokenize(options: {
    text: string;
    addSpecial?: boolean;
    parseSpecial?: boolean;
    withPieces?: boolean;
    signal?: AbortSignal;
  }): Promise<number[] | ApiTokenizePiece[]> {
    const body = objectToSnakeCase({
      content: options.text,
      addSpecial: options.addSpecial,
      parseSpecial: options.parseSpecial,
      withPieces: options.withPieces,
      model: this.id,
    });

    const response = await requestJson<ApiTokenizeResponse>({
      fetchFn: this.client.clientOptions.fetchFn,
      baseUrl: this.client.BASE_URL,
      method: "POST",
      pathName: "/tokenize",
      body,
      signal: options.signal,
    });

    return response.tokens as number[] | ApiTokenizePiece[];
  }

  /**
   * Detokenizes token IDs back into text.
   * @param tokens Array of token IDs to detokenize.
   * @param signal Optional abort signal.
   * @returns The detokenized text.
   */
  public async detokenize(tokens: number[], signal?: AbortSignal): Promise<string> {
    const body = objectToSnakeCase({ tokens, model: this.id });

    const response = await requestJson<ApiDetokenizeResponse>({
      fetchFn: this.client.clientOptions.fetchFn,
      baseUrl: this.client.BASE_URL,
      method: "POST",
      pathName: "/detokenize",
      body,
      signal,
    });

    return response.content;
  }
}

/**
 * Configuration options for creating a Client instance.
 */
export interface ClientOptions {
  /**
   * Optional custom fetch function.
   *
   * Use this to inject a custom HTTP client (e.g., for testing, logging,
   * or polyfilling fetch in different environments). Defaults to the global
   * `fetch` function if not provided.
   *
   * @example
   * ```typescript
   * const client = await Client.from("http://localhost:8080", {
   *   fetchFn: myCustomFetch,
   * });
   * ```
   */
  fetchFn: FetchFn;
}

/**
 * Thrown when loading a model fails.
 */
export class ModelLoadError extends Error {
  public readonly inner?: unknown;

  constructor(details: string, inner?: unknown) {
    super(details);
    this.name = "ModelLoadError";
    this.inner = inner;
  }
}

/**
 * Thrown when unloading a model fails.
 */
export class ModelUnloadError extends Error {
  public readonly inner?: unknown;

  constructor(details: string, inner?: unknown) {
    super(details);
    this.name = "ModelUnloadError";
    this.inner = inner;
  }
}

/**
 * Thrown when a model ID is invalid or not available.
 */
export class InvalidModelError extends Error {
  public readonly modelId: string;

  constructor(modelId: string, details: string) {
    super(details);
    this.name = "InvalidModelError";
    this.modelId = modelId;
  }
}

/**
 * Represents the current status of a model in the router.
 */
export type ModelStatus = "loaded" | "unloading" | "loading" | "unloaded" | "failed";

/**
 * Represents the managed state of a model within the Client.
 */
export interface ManagedModel {
  id: string;
  status: ModelStatus;
  inCache: boolean;
}

/**
 * Represents the HTTP Client connecting to the inference API server.
 */
export class Client {
  public readonly BASE_URL: URL;

  /** Configuration options for this Client instance. */
  public readonly clientOptions: ClientOptions;

  /** Internal tracking of model statuses. */
  public modelStatuses: Map<string, ManagedModel>;

  private constructor(baseUrl: URL, clientOptions: ClientOptions) {
    this.BASE_URL = baseUrl;
    this.clientOptions = clientOptions;
    this.modelStatuses = new Map();
  }

  /**
   * Fetches the current model statuses from the /models endpoint.
   * @param baseUrl The base URL of the inference server.
   * @param clientOptions Configuration options for the client.
   * @returns The Map of model statuses.
   * @throws ApiRequestError when the API request fails.
   */
  private static async fetchModelStatuses(
    baseUrl: URL,
    fetchFn: FetchFn,
  ): Promise<Map<string, ManagedModel>> {
    const response = await requestJson<ApiModelsResponse>({
      fetchFn,
      baseUrl,
      method: "GET",
      pathName: "/models",
      transformBody: (body) => objectToCamelCase(body, true),
    });

    const statuses = new Map<string, ManagedModel>();
    for (const entry of response.data) {
      statuses.set(entry.id, {
        id: entry.id,
        status: entry.status?.value ?? "unloaded",
        inCache: entry.inCache,
      });
    }
    return statuses;
  }

  /**
   * Polls the /models endpoint until the model reaches the expected status.
   * Updates local cache on every successful poll to reflect real-time server state.
   * @param modelId The model identifier to poll.
   * @param expectedStatus The status to wait for.
   * @param maxAttempts Maximum number of polling attempts.
   * @param delayMs Delay between polling attempts.
   * @returns The final model status.
   * @throws UnexpectedServerBehaviorError when the model fails to reach expected status.
   */
  private async pollForModelStatus(
    modelId: string,
    expectedStatus: ModelStatus,
    maxAttempts: number = 60,
    delayMs: number = 1000,
  ): Promise<ModelStatus> {
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      const statuses = await Client.fetchModelStatuses(
        new URL(this.BASE_URL.toString()),
        this.clientOptions.fetchFn,
      );

      const modelStatus = statuses.get(modelId);

      if (modelStatus) {
        this.modelStatuses.set(modelId, modelStatus);
      }

      const status = modelStatus?.status;

      if (status === "failed") {
        throw new UnexpectedServerBehaviorError(`Model "${modelId}" server reported failed status`);
      }

      if (status === expectedStatus) {
        return status;
      }

      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }

    throw new UnexpectedServerBehaviorError(
      `Model "${modelId}" did not reach expected status "${expectedStatus}" after ${maxAttempts} attempts`,
    );
  }

  /**
   * Loads a model into the inference router.
   * Polls until the model reaches "loaded" status.
   * @param id The model identifier to load.
   * @throws ModelLoadError when the model fails to load.
   */
  public async load(id: string): Promise<void> {
    const currentStatus = this.modelStatuses.get(id);
    if (currentStatus?.status === "loaded") {
      return;
    }

    let result: ApiModelLoadUnloadResponse;
    try {
      result = await requestJson<ApiModelLoadUnloadResponse>({
        fetchFn: this.clientOptions.fetchFn,
        baseUrl: this.BASE_URL,
        method: "POST",
        pathName: "/models/load",
        body: { model: id },
        transformBody: (body) => objectToCamelCase(body, true),
      });
    } catch (error) {
      if (error instanceof ApiRequestError && error.httpStatusCode === 400) {
        if (error.responseBody?.includes("model is already loaded")) {
          this.modelStatuses.set(id, {
            id,
            status: "loaded",
            inCache: true,
          });
          return;
        }
      }
      throw new ModelLoadError("Model load request failed", error);
    }

    this.modelStatuses.set(id, {
      id,
      status: "loading",
      inCache: true,
    });

    try {
      await this.pollForModelStatus(id, "loaded", 60, 1000);
      this.modelStatuses.set(id, {
        id,
        status: "loaded",
        inCache: true,
      });
    } catch (error) {
      this.modelStatuses.set(id, {
        id,
        status: "failed",
        inCache: true,
      });
      throw new ModelLoadError("The model failed to load.", error);
    }
  }

  /**
   * Unloads a model from the inference router.
   * Polls until the model reaches "unloaded" or "unloading" status.
   * @param id The model identifier to unload.
   * @throws ModelUnloadError when the model fails to unload.
   */
  public async unload(id: string): Promise<void> {
    let result: ApiModelLoadUnloadResponse;
    try {
      result = await requestJson<ApiModelLoadUnloadResponse>({
        fetchFn: this.clientOptions.fetchFn,
        baseUrl: this.BASE_URL,
        method: "POST",
        pathName: "/models/unload",
        body: { model: id },
        transformBody: (body) => objectToCamelCase(body, true),
      });
    } catch (error) {
      if (error instanceof ApiRequestError && error.httpStatusCode === 400) {
        if (error.responseBody?.includes("model is not loaded")) {
          this.modelStatuses.set(id, {
            id,
            status: "unloaded",
            inCache: false,
          });
          return;
        }
      }
      throw new ModelUnloadError("Model unload request failed", error);
    }

    this.modelStatuses.set(id, {
      id,
      status: "unloading",
      inCache: false,
    });

    try {
      await this.pollForModelStatus(id, "unloaded", 30, 1000);
      this.modelStatuses.set(id, {
        id,
        status: "unloaded",
        inCache: false,
      });
    } catch (error) {
      this.modelStatuses.set(id, {
        id,
        status: "failed",
        inCache: false,
      });
      throw new ModelUnloadError("The model failed to unload.", error);
    }
  }

  /**
   * Checks if this model is currently loaded in the router.
   * @param id The model identifier.
   * @returns true if loaded, false if unloaded.
   * @throws ModelLoadError if the model is not found in the client cache.
   */
  public async isModelLoaded(id: string): Promise<boolean> {
    const model = this.modelStatuses.get(id);
    if (!model) {
      throw new ModelLoadError(
        `Model "${id}" not found in client cache. Available models: ${[...this.modelStatuses.keys()].join(", ")}`,
      );
    }

    return model.status === "loaded";
  }

  /**
   * Gets the current status of a model.
   * @param id The model identifier.
   * @returns The model status.
   * @throws InvalidModelError if the model is not tracked.
   */
  public getModelStatus(id: string): ManagedModel {
    const model = this.modelStatuses.get(id);
    if (!model) {
      throw new InvalidModelError(id, `Model "${id}" not found in client cache`);
    }
    return model;
  }

  /**
   * Creates a new `TextModel` instance associated with this client.
   * @param id The model identifier to fetch metadata for.
   * @returns The `TextModel` instance.
   * @throws InvalidModelError if the model is not tracked or not loaded.
   */
  public async createTextModel(id: string): Promise<TextModel> {
    const modelStatus = this.modelStatuses.get(id);
    if (!modelStatus) {
      throw new InvalidModelError(
        id,
        `Model "${id}" is not tracked by the client. Available models: ${[...this.modelStatuses.keys()].join(", ")}`,
      );
    }

    if (modelStatus.status !== "loaded") {
      throw new InvalidModelError(
        id,
        `Model "${id}" is not currently loaded. Current status: ${modelStatus.status}`,
      );
    }

    return TextModel.from(this, id);
  }

  /**
   * Creates a new `EmbeddingModel` instance associated with this client.
   * @param id The model identifier to fetch metadata for.
   * @returns The `EmbeddingModel` instance.
   * @throws InvalidModelError if the model is not tracked or not loaded.
   */
  public async createEmbeddingModel(id: string): Promise<EmbeddingModel> {
    const modelStatus = this.modelStatuses.get(id);
    if (!modelStatus) {
      throw new InvalidModelError(
        id,
        `Model "${id}" is not tracked by the client. Available models: ${[...this.modelStatuses.keys()].join(", ")}`,
      );
    }

    if (modelStatus.status !== "loaded") {
      throw new InvalidModelError(
        id,
        `Model "${id}" is not currently loaded. Current status: ${modelStatus.status}`,
      );
    }

    return EmbeddingModel.from(this, id);
  }

  /**
   * Static factory method to create a Client instance.
   * Queries the /models endpoint to initialize model tracking.
   * @param baseUrl The base URL of the inference server.
   * @param options Optional configuration options.
   * @returns The `Client` instance.
   * @throws ApiRequestError when the API request fails.
   */
  public static async from(
    baseUrl: string | URL,
    options?: Partial<ClientOptions>,
  ): Promise<Client> {
    const clientOptions: ClientOptions = {
      fetchFn: globalThis["fetch"],
      ...options,
    };

    baseUrl = new URL(baseUrl);

    const statuses = await Client.fetchModelStatuses(baseUrl, clientOptions.fetchFn);

    const client = new Client(baseUrl, clientOptions);
    client.modelStatuses = statuses;

    return client;
  }

  /**
   * Refreshes the model status cache from the server.
   * @throws ApiRequestError when the API request fails.
   */
  public async refreshModelStatuses(): Promise<void> {
    this.modelStatuses = await Client.fetchModelStatuses(this.BASE_URL, this.clientOptions.fetchFn);
  }
}
