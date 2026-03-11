import { toJsonSchema } from "@valibot/to-json-schema";
import { err, ok, Result } from "neverthrow";
import * as v from "valibot";
import {
  requestJson,
  requestStream,
  type ApiChatCompletionOptions,
  type ApiChatCompletionStreamChunk,
  type ApiCompletionStreamChunk,
  type ApiModelEntry,
  type ApiModelLoadUnloadResponse,
  type ApiModelsResponse,
  type ApiPropsResponse,
  type ApiRequestError,
  type ApiTimingMetrics,
} from "./api.js";
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
 * Represents an error caused by invalid parameters passed to the client.
 */
export interface InvalidParameterError {
  kind: "InvalidParameter";
  /** Detailed message explaining the validation failure. */
  details: string;
}

/**
 * Represents an error caused by an empty message history.
 */
export interface EmptyMessageArrayError {
  kind: "EmptyMessageHistory";
  /** Explanation of why the history was deemed empty. */
  details: string;
}

/**
 * Represents an error caused when a request is aborted by the client.
 */
export interface AbortedRequestError {
  kind: "RequestAborted";
  /** Description of the cancellation reason. */
  details: string;
}

/**
 * Represents an unexpected error originating from the server.
 */
export interface UnexpectedServerBehaviorError {
  kind: "ServerError";
  /** Description of the unexpected behavior. */
  details: string;
  /** Optional underlying cause/error object. */
  cause?: unknown;
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
   * @returns A `Result` containing the model instance or a `ApiRequestError`.
   */
  public static async from(
    client: Client,
    id: string,
  ): Promise<Result<TextModel, ApiRequestError>> {
    const urlWithModel = new URL(client.BASE_URL);
    urlWithModel.searchParams.set("model", id);

    const queryResult = await requestJson<ApiPropsResponse>({
      baseUrl: urlWithModel,
      method: "GET",
      pathName: "/props",
      transformBody: (body) => objectToCamelCase(body, true),
    });

    if (queryResult.isErr()) {
      return err({
        ...queryResult.error,
        kind: "RequestError" as const,
        details: "Failed to query model properties: " + queryResult.error,
      } satisfies ApiRequestError);
    }

    const response = queryResult.value;

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

    return ok(new TextModel(client, id, { totalSlots, contextSize: nCtx }));
  }

  /**
   * Generates a text completion stream for a single prompt.
   *
   * Suitable for tasks that do not require conversation history (e.g., completion, translation).
   * Also supports code infilling when the input is an object with prefix/suffix.
   * @param options Configuration for the prediction including prompt, sampling settings, and callbacks.
   * @returns A `Result` containing the completion output and metrics, or a `CompletionError`.
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
  }): Promise<Result<PredictResult, ApiRequestError | CompletionError>> {
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

      const requestResult = await requestStream<ApiCompletionStreamChunk>({
        baseUrl: this.client.BASE_URL,
        method: "POST",
        // select native endpoint: /completion for standard text, /infill for infilling
        pathName: typeof options.input === "string" ? "/completion" : "/infill",
        body,
        signal: options.signal,
      });

      if (requestResult.isErr()) {
        return err(requestResult.error);
      }

      let content = "";

      for await (const chunk of requestResult.value) {
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

          return ok({
            content,
            timings: chunk.timings!,
            finishReason: finishReason!,
          });
        }
      }

      return err({
        kind: "ServerError",
        details: "The inference server did not provide a completion termination signal",
      });
    } catch (error) {
      if (error instanceof DOMException && error.code === error.ABORT_ERR) {
        return err({
          kind: "RequestAborted",
          details: "The completion request was cancelled before completion",
        });
      } else {
        return err({
          kind: "ServerError",
          details: "An unexpected error occurred during inference",
          cause: error,
        });
      }
    }
  }

  /**
   * Generates a chat response based on conversation history and optional tools.
   *
   * Handles tool calls, reasoning content, and multi-turn context.
   * @param options Configuration including instructions, history, sampling, and tools.
   * @returns A `Result` containing the response message and finish reason, or a `ChatCompletionError`.
   */
  public async respond(options: {
    instructions: string;
    history: History;
    sampling: SamplingResult;
    toolset?: ToolsetResult;
    maxTokens?: number;
    signal?: AbortSignal;
    onEvent?: ((data: ChatCompletionCallbackEvent) => void) | undefined;
  }): Promise<
    Result<RespondResult, InvalidParameterError | ApiRequestError | ChatCompletionError>
  > {
    if (options.sampling.grammar !== undefined && options.toolset !== undefined) {
      return err({
        kind: "InvalidParameter",
        details: "Grammar-based sampling (JSON or GBNF) and tool calling are mutually exclusive",
      });
    } else if (options.history.length === 0) {
      return err({
        kind: "InvalidParameter",
        details: "Conversation history is required for response generation",
      });
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

      const requestResult = await requestStream<ApiChatCompletionStreamChunk>({
        baseUrl: this.client.BASE_URL,
        method: "POST",
        pathName: "/v1/chat/completions",
        body,
        signal: options.signal,
      });

      if (requestResult.isErr()) {
        return err(requestResult.error);
      }

      const collected = {
        role: "",
        reasoningContent: "",
        content: "",
        toolCalls: [] as { id: string; name: string; arguments: string }[],
      };

      for await (const chunk of requestResult.value) {
        const data = chunk.choices[0]!;

        if (data.delta.role) {
          // got message role
          collected.role += data.delta.role;
        }

        if (data.delta.content) {
          // got message content fragment
          collected.content += data.delta.content;
          options.onEvent?.({
            kind: "ContentToken",
            fragment: data.delta.content,
            timings: chunk.timings!,
          });
        }

        if (data.delta.reasoningContent) {
          // got message reasoning fragment
          collected.reasoningContent += data.delta.reasoningContent;
          options.onEvent?.({
            kind: "ReasoningToken",
            fragment: data.delta.reasoningContent,
            timings: chunk.timings!,
          });
        }

        if (data.delta.toolCalls) {
          // got toolcall fragment
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
          // finished generation
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

          return ok({
            response: generatedMessage,
            timings: chunk.timings!,
            finishReason: finishReason,
          });
        }
      }

      return err({
        kind: "ServerError",
        details: "The inference server did not provide a completion termination signal",
      });
    } catch (error) {
      if (error instanceof DOMException && error.code === error.ABORT_ERR) {
        return err({
          kind: "RequestAborted",
          details: "The completion request was cancelled before completion",
        });
      } else {
        return err({
          kind: "ServerError",
          details: "An unexpected error occurred during inference",
          cause: error,
        });
      }
    }
  }

  /**
   * Executes an autonomous agent loop.
   *
   * Repeatedly calls `respond` and executes returned tool calls until no tools are requested.
   * @param options Configuration including instructions, history, tools, and callbacks.
   * @returns A `Result` containing the final chat history or an error.
   */
  public async act(options: {
    instructions: string;
    history: History;
    sampling: SamplingResult;
    toolset: ToolsetResult;
    signal?: AbortSignal;
    onEvent?: (data: ChatCompletionCallbackEvent) => void;
  }) {
    const threadHistory: History = [];

    // agent loop
    while (true) {
      const responseResult = await this.respond({
        instructions: options.instructions,
        history: [...options.history, ...threadHistory],
        sampling: options.sampling,
        toolset: options.toolset,
        signal: options.signal!,
        onEvent: options.onEvent,
        maxTokens: Infinity,
      });

      if (responseResult.isErr()) {
        return responseResult;
      }

      const { finishReason, response } = responseResult.value;

      // add assistant response to history
      threadHistory.push(response);

      if (finishReason !== "ToolCallRequest") {
        break;
      }

      // execute tools in parallel
      const toolResults = await Promise.all(
        response.requestedToolCalls.map(executeTool.bind(null, options.toolset)),
      );

      threadHistory.push(...toolResults);
    }

    return ok(threadHistory);
  }
}

/**
 * Represents an error caused when loading a model fails.
 */
export interface ModelLoadError {
  kind: "ModelLoadError";
  /** Description of the loading failure. */
  details: string;
}

/**
 * Represents an error caused when unloading a model fails.
 */
export interface ModelUnloadError {
  kind: "ModelUnloadError";
  /** Description of the unloading failure. */
  details: string;
}

/**
 * Represents an error caused when a model ID is invalid or not available.
 */
export interface InvalidModelError {
  kind: "InvalidModel";
  /** The invalid model ID that was provided. */
  modelId: string;
  /** Explanation of why the model is invalid. */
  details: string;
}

/**
 * Represents the HTTP Client connecting to the inference API server.
 */
export class Client {
  public readonly BASE_URL: URL;

  /** Cached list of available models from the router. */
  public readonly availableModels: Map<string, ApiModelEntry>;

  private constructor(baseUrl: string | URL, availableModels: Map<string, ApiModelEntry>) {
    this.BASE_URL = new URL(baseUrl);
    this.availableModels = availableModels;
  }

  /**
   * Loads a model into the inference router.
   * @param id The model identifier to load.
   * @returns A Result containing success status or error.
   */
  public async load(id: string): Promise<Result<void, ModelLoadError | ApiRequestError>> {
    const result = await requestJson<ApiModelLoadUnloadResponse>({
      baseUrl: this.BASE_URL,
      method: "POST",
      pathName: "/models/load",
      body: { model: id },
      transformBody: (body) => objectToCamelCase(body, true),
    });

    if (result.isErr()) {
      return err(result.error);
    }

    if (!result.value.success) {
      return err({
        kind: "ModelLoadError",
        details: "Model load request returned non-success response",
      });
    }

    return ok();
  }

  /**
   * Unloads a model from the inference router.
   * @param id The model identifier to unload.
   * @returns A Result containing success status or error.
   */
  public async unload(id: string): Promise<Result<void, ModelUnloadError | ApiRequestError>> {
    const result = await requestJson<ApiModelLoadUnloadResponse>({
      baseUrl: this.BASE_URL,
      method: "POST",
      pathName: "/models/unload",
      body: { model: id },
      transformBody: (body) => objectToCamelCase(body, true),
    });

    if (result.isErr()) {
      return err(result.error);
    }

    if (!result.value.success) {
      return err({
        kind: "ModelUnloadError",
        details: "Model unload request returned non-success response",
      });
    }

    return ok(undefined);
  }

  /**
   * Checks if this model is currently loaded in the router.
   * @returns A Result containing true if loaded, false if unloaded, or an error.
   */
  public async isModelLoaded(
    id: string,
  ): Promise<Result<boolean, ApiRequestError | ModelLoadError>> {
    const result = await requestJson<ApiModelsResponse>({
      baseUrl: this.BASE_URL,
      method: "GET",
      pathName: "/models",
      transformBody: (body) => objectToCamelCase(body, true),
    });

    if (result.isErr()) {
      return err(result.error);
    }

    const modelEntry = result.value.data.find((entry) => entry.id === id);

    if (!modelEntry) {
      return err({
        kind: "ModelLoadError",
        details: `Model "${id}" not found in router`,
      });
    }

    if (!modelEntry.status) {
      return err({
        kind: "ModelLoadError",
        details: `Model "${id}" has no status information`,
      });
    }

    const isLoaded = modelEntry.status.value === "loaded";

    return ok(isLoaded);
  }

  /**
   * Creates a new `TextModel` instance associated with this client.
   * @param id The model identifier to fetch metadata for.
   * @returns A promise resolving to a `Result` containing the `TextModel`.
   */
  public async createTextModel(id: string) {
    // Validate model exists in cache
    if (!this.availableModels.has(id)) {
      return err({
        kind: "InvalidModel",
        modelId: id,
        details: `Model "${id}" is not available in the router. Available models: ${[...this.availableModels.keys()].join(", ")}`,
      });
    }

    return TextModel.from(this, id);
  }

  /**
   * Static factory method to create a Client instance.
   * Queries the /models endpoint to cache available models.
   * @param baseUrl The base URL of the inference server.
   * @returns A promise resolving to a Result containing the Client or an error.
   */
  public static async from(baseUrl: string): Promise<Result<Client, ApiRequestError>> {
    const result = await requestJson<ApiModelsResponse>({
      baseUrl: new URL(baseUrl),
      method: "GET",
      pathName: "/models",
    });

    if (result.isErr()) {
      return err(result.error);
    }

    const models = new Map(result.value.data.map((entry) => [entry.id, entry]));
    const client = new Client(baseUrl, models);

    return ok(client);
  }
}
