import { err, ok, type Result } from "neverthrow";
import { apiStreamChunker } from "./streamChunker.js";

/**
 * Represents a model's status in the router.
 */
export interface ApiModelStatus {
  value: "loaded" | "unloading" | "loading" | "unloaded" | "failed";
  args?: string[];
  failed?: boolean;
  exitCode?: number;
}

/**
 * Represents a model entry from the /models endpoint.
 */
export interface ApiModelEntry {
  id: string;
  inCache: boolean;
  path?: string;
  status?: ApiModelStatus;
  loaded?: boolean;
}

/**
 * Response from GET /models endpoint.
 */
export interface ApiModelsResponse {
  data: ApiModelEntry[];
}

/**
 * Request for model load/unload operations.
 */
export interface ApiModelLoadUnloadRequest {
  model: string;
}

/**
 * Response for model load/unload operations.
 */
export interface ApiModelLoadUnloadResponse {
  success: boolean;
}

/**
 * Return value of llama-server's `GET /props` endpoint (build b8157-2943210c1).
 */
export interface ApiPropsResponse {
  defaultGenerationSettings: {
    params: {
      seed: number;
      temperature: number;
      dynatempRange: number;
      dynatempExponent: number;
      topK: number;
      topP: number;
      minP: number;
      topNSigma: number;
      xtcProbability: number;
      xtcThreshold: number;
      typicalP: number;
      repeatLastN: number;
      repeatPenalty: number;
      presencePenalty: number;
      frequencyPenalty: number;
      dryMultiplier: number;
      dryBase: number;
      dryAllowedLength: number;
      dryPenaltyLastN: number;
      mirostat: number;
      mirostatTau: number;
      mirostatEta: number;
      maxTokens: number;
      nPredict: number;
      nKeep: number;
      nDiscard: number;
      ignoreEos: boolean;
      stream: boolean;
      nProbs: number;
      minKeep: number;
      chatFormat: string;
      reasoningFormat: string;
      reasoningInContent: boolean;
      thinkingForcedOpen: boolean;
      samplers: string[];
      speculativeNMax: number;
      speculativeNMin: number;
      speculativePMin: number;
      speculativeType: string;
      speculativeNgramSizeN: number;
      speculativeNgramSizeM: number;
      speculativeNgramMHits: number;
      timingsPerToken: boolean;
      postSamplingProbs: boolean;
      backendSampling: boolean;
      lora: string[];
    };
    nCtx: number;
  };
  totalSlots: number;
  modelAlias: string;
  modelPath: string;
  modalities: {
    vision: boolean;
    audio: boolean;
  };
  endpointSlots: boolean;
  endpointProps: boolean;
  endpointMetrics: boolean;
  webUi: boolean;
  webUiSettings: Record<string, unknown>;
  chatTemplate: string;
  chatTemplateCaps: {
    supportsParallelToolCalls: boolean;
    supportsPreserveReasoning: boolean;
    supportsStringContent: boolean;
    supportsSystemRole: boolean;
    supportsToolCalls: boolean;
    supportsTools: boolean;
    supportsTypedContent: boolean;
  };
  bosToken: string;
  eosToken: string;
  buildInfo: string;
  isSleeping: boolean;
}

export interface ApiTimingMetrics {
  promptN: number;
  promptMs: number;
  promptPerTokenMs: number;
  promptPerSecond: number;
  predictedN: number;
  predictedMs: number;
  predictedPerTokenMs: number;
  predictedPerSecond: number;
}

export interface ApiUsageMetrics {
  completionTokens: number;
  promptTokens: number;
  totalTokens: number;
}

export interface ApiBaseCompletionOptions {
  model: string;
  temperature?: number;
  frequencyPenalty?: number;
  logitBias?: Record<string, number>;
  maxTokens?: number;
  presencePenalty?: number;
  topP?: number;
  seed?: number;
  stop?: string[];
  stream?: boolean;
  /* llama.cpp extended parameters */
  topK?: number;
  minP?: number;
  typicalP?: number;
  repeatPenalty?: number;
  repeatLastN?: number;
  dryMultiplier?: number;
  dryBase?: number;
  dryAllowedLength?: number;
  dryPenaltyLastN?: number;
  drySequenceBreakers?: string[];
  xtcProbability?: number;
  xtcThreshold?: number;
  mirostat?: 0 | 1 | 2;
  mirostatTau?: number;
  mirostatEta?: number;
  dynatempRange?: number;
  dynatempExponent?: number;
  cachePrompt?: boolean;
  grammar?: string;
  jsonSchema?: object;
  ignoreEos?: boolean;
}

export interface ApiCompletionOptions extends ApiBaseCompletionOptions {
  prompt: string;
}

/**
 * Represents a stream chunk from llama-server's native `/completion` and `/infill` endpoints.
 *
 * These endpoints differ from the OpenAI-compatible `/v1/completions` format.
 * This interface matches the response format of llama-server (llama.cpp) b8157.
 */
export interface ApiCompletionStreamChunk {
  index: number;
  content: string;
  tokens: number[];
  stop: boolean;
  idSlot: number;
  tokensPredicted: number;
  tokensEvaluated: number;

  // fields only present in the final chunk
  model?: string;
  generationSettings?: {};
  prompt?: string;
  hasNewLine?: boolean;
  truncated?: boolean;
  stopType?: "none" | "eos" | "limit" | "word";
  stoppingWord?: string;
  tokensCached?: number;
  timings?: ApiTimingMetrics;
}

export interface ApiMessage {
  role: string;
  content: string;
  reasoningContent?: string;
  prefill?: boolean;
}

export interface ApiChatCompletionOptions extends ApiBaseCompletionOptions {
  messages: ApiMessage[];
  reasoningFormat?: "none" | "auto" | "deepseek" | "deepseek-legacy";
}

export interface ApiChatCompletionStreamChunk {
  id: string;
  object: "chat.completion.chunk";
  systemFingerprint: string;
  model: string;
  created: number;
  choices: {
    delta: {
      role?: string;
      content?: string;
      reasoningContent?: string;
      toolCalls?: {
        index: number;
        id?: string;
        type?: "function";
        function: { name?: string; arguments?: string };
      }[];
    };
    index: number;
    logprobs: null;
    finishReason: "stop" | "length" | "content_filter" | "tool_calls" | null;
  }[];
  usage?: ApiUsageMetrics;
  timings?: ApiTimingMetrics;
}

/**
 * Represents an HTTP-level API request failure.
 */
export interface ApiRequestError {
  kind: "RequestError";
  /** General description of the failure. */
  details: string;
  /** The HTTP status code returned by the server. */
  httpStatusCode: number;
  /** Optional raw response body if available. */
  responseBody?: string;
}

/**
 * Normalizes URL paths by combining a base URL with a path name.
 *
 * Ensures that trailing slashes from the base and leading slashes from the path
 * are handled correctly to avoid malformed URLs (e.g., "base//path").
 * @param baseUrl The base URL object.
 * @param pathName The path string to append.
 * @returns A new URL with the combined and normalized path.
 */
const combinePathNames = (baseUrl: URL, pathName: string): URL => {
  const base = baseUrl.pathname.replace(/\/$/, "");
  const path = pathName.replace(/^\/+/, "");
  const newUrl = new URL(baseUrl);
  newUrl.pathname = `${base}/${path}`;
  return newUrl;
};

/**
 * Performs a JSON-based API request (GET or POST).
 *
 * Handles response parsing and error mapping to `ApiRequestError`.
 * @param options Configuration for the request including URL, method, body, and abort signal.
 * @returns A `Result` containing the parsed JSON or an `ApiRequestError`.
 */
export async function requestJson<J extends object>(options: {
  fetchFn: typeof fetch;
  baseUrl: URL;
  method: "GET" | "POST";
  pathName: string;
  body?: object;
  signal?: AbortSignal | undefined | null;
  transformBody?: (input: object) => object;
}): Promise<Result<J, ApiRequestError>> {
  const requestUrl = combinePathNames(options.baseUrl, options.pathName);

  const response = await options.fetchFn(requestUrl, {
    method: options.method,
    headers: { "Content-Type": "application/json" },
    signal: options.signal ?? null,
    body: options.method === "GET" ? null : JSON.stringify(options.body),
  });

  if (!response.ok) {
    return err({
      kind: "RequestError",
      details: "The API call failed",
      httpStatusCode: response.status,
      responseBody: await response.text(),
    });
  }

  let json = await response.json();

  if (options.transformBody) {
    json = options.transformBody(json as object);
  }

  return ok(json as J);
}

/**
 * Performs an API request that returns a Server-Sent Events (SSE) stream.
 *
 * Wraps the stream in `apiStreamChunker` for JSON parsing.
 * @param options Configuration for the stream request.
 * @returns A `Result` containing the async generator stream or an `ApiRequestError`.
 */
export async function requestStream<C extends object>(options: {
  fetchFn: typeof fetch;
  baseUrl: URL;
  method: "GET" | "POST";
  pathName: string;
  body: object;
  signal: AbortSignal | undefined | null;
}): Promise<Result<AsyncGenerator<C, void, unknown>, ApiRequestError>> {
  const requestUrl = combinePathNames(options.baseUrl, options.pathName);

  const response = await options.fetchFn(requestUrl, {
    method: options.method,
    headers: { "Content-Type": "application/json" },
    signal: options.signal ?? null,
    body: JSON.stringify(options.body),
  });

  if (!response.ok) {
    return err({
      kind: "RequestError",
      details: "The API call failed",
      httpStatusCode: response.status,
      responseBody: await response.text(),
    });
  }

  const stream = apiStreamChunker<C>(response.body!);

  return ok(stream);
}
