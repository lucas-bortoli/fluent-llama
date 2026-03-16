import { err, ok, type Result } from "neverthrow";
import {
  type ApiEmbeddingResponse,
  type ApiRequestError,
  requestJson,
  type ApiTimingMetrics,
} from "./api.js";
import { objectToSnakeCase } from "./helpers.js";
import {
  type AbortedRequestError,
  type Client,
  type CompletionError,
  type UnexpectedServerBehaviorError,
} from "./client.js";

/**
 * Embedding normalization options.
 * - NoNormalization: No normalization (or null)
 * - MaxAbs: Max absolute normalization
 * - L1: Taxicab normalization (L1)
 * - L2: Euclidean normalization (L2)
 */
export type EmbeddingNormalization = "NoNormalization" | "MaxAbs" | "L1" | "L2";

/**
 * Result structure returned from embedding generation.
 * Returns an array of embedding vectors (one per input).
 */
export type EmbeddingResult = number[][];

/**
 * Result structure for a single embedding.
 * Returns a single embedding vector.
 */
export type SingleEmbeddingResult = number[];

/**
 * Input for EmbeddingModel.embed() method.
 * Supports text-only strings, arrays of text, or multimodal (image + text) inputs.
 */
export type EmbeddingInput =
  /** Single text input → returns single embedding */
  | string
  /** Multiple text inputs → returns array of embeddings */
  | string[]
  /** Multimodal input with image data → returns array of embeddings */
  | {
      /** Text prompt for the multimodal input */
      prompt: string;
      /** Base64-encoded image data URIs */
      multimodalData: string[];
    };

/**
 * Represents an embedding model instance connected to the inference API.
 */
export class EmbeddingModel {
  public readonly client: Client;
  public readonly id: string;

  private constructor(client: Client, id: string) {
    this.client = client;
    this.id = id;
  }

  /**
   * Constructs an `EmbeddingModel` instance.
   */
  public static async from(
    client: Client,
    id: string,
  ): Promise<Result<EmbeddingModel, ApiRequestError>> {
    return ok(new EmbeddingModel(client, id));
  }

  /**
   * Generates embeddings for text or multimodal input.
   *
   * @param options - Embedding options
   * @param options.input - Text string (returns single embedding) or array of texts/multimodal input (returns array of embeddings)
   * @param options.normalization - Normalization method: "NoNormalization", "MaxAbs", "L1", or "L2". Defaults to null.
   * @param options.signal - Optional AbortSignal to cancel the request
   * @returns Promise resolving to single embedding vector (number[]) for single text input, or array of embedding vectors (number[][]) for multiple inputs
   *
   * @example
   * // Single text → returns number[]
   * const embedding = await model.embed({ input: "hello" });
   *
   * // Multiple texts → returns number[][]
   * const embeddings = await model.embed({ input: ["hello", "world"] });
   *
   * // Multimodal → returns number[][]
   * const embeddings = await model.embed({
   *   input: { prompt: "describe", multimodalData: ["data:image/png;base64,..."] }
   * });
   */
  public embed(options: {
    input: string;
    normalization?: EmbeddingNormalization | null;
    signal?: AbortSignal;
  }): Promise<Result<SingleEmbeddingResult, ApiRequestError | CompletionError>>;
  public embed(options: {
    input: string[] | { prompt: string; multimodalData: string[] };
    normalization?: EmbeddingNormalization | null;
    signal?: AbortSignal;
  }): Promise<Result<EmbeddingResult, ApiRequestError | CompletionError>>;
  public async embed(options: {
    input: EmbeddingInput;
    normalization?: EmbeddingNormalization | null;
    signal?: AbortSignal;
  }): Promise<Result<EmbeddingResult | SingleEmbeddingResult, ApiRequestError | CompletionError>> {
    try {
      const content =
        typeof options.input === "string" || Array.isArray(options.input)
          ? options.input
          : options.input;

      const body = objectToSnakeCase({
        content,
        model: this.id,
        ...(options.normalization !== undefined &&
          options.normalization !== null && { normalization: options.normalization }),
      });

      const requestResult = await requestJson<ApiEmbeddingResponse>({
        fetchFn: this.client.clientOptions.fetchFn,
        baseUrl: this.client.BASE_URL,
        method: "POST",
        pathName: "/embedding",
        body,
        signal: options.signal,
      });

      if (requestResult.isErr()) {
        return err(requestResult.error);
      }

      const embeddings = requestResult.value.map((item) =>
        Array.isArray(item.embedding[0]) ? item.embedding[0] : item.embedding,
      );

      if (typeof options.input === "string") {
        return ok(embeddings[0]!);
      }

      return ok(embeddings);
    } catch (error) {
      if (error instanceof DOMException && error.code === error.ABORT_ERR) {
        return err({
          kind: "RequestAborted",
          details: "The embedding request was cancelled before completion",
        } satisfies AbortedRequestError);
      } else {
        return err({
          kind: "ServerError",
          details: "An unexpected error occurred during embedding generation",
          cause: error,
        } satisfies UnexpectedServerBehaviorError);
      }
    }
  }
}
