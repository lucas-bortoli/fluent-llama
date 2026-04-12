import {
  type ApiEmbeddingResponse,
  type ApiRequestError,
  requestJson,
  type ApiTimingMetrics,
} from "./api.js";
import { objectToSnakeCase } from "./helpers.js";
import {
  AbortedRequestError,
  type Client,
  type CompletionError,
  UnexpectedServerBehaviorError,
} from "./client.js";

export type EmbeddingNormalization = "NoNormalization" | "MaxAbs" | "L1" | "L2";

export type EmbeddingResult = number[][];

export type SingleEmbeddingResult = number[];

export type EmbeddingInput =
  | string
  | string[]
  | {
      prompt: string;
      multimodalData: string[];
    };

export class EmbeddingModel {
  public readonly client: Client;
  public readonly id: string;

  private constructor(client: Client, id: string) {
    this.client = client;
    this.id = id;
  }

  public static async from(client: Client, id: string): Promise<EmbeddingModel> {
    return new EmbeddingModel(client, id);
  }

  public embed(options: {
    input: string;
    normalization?: EmbeddingNormalization | null;
    signal?: AbortSignal;
  }): Promise<SingleEmbeddingResult>;
  public embed(options: {
    input: string[] | { prompt: string; multimodalData: string[] };
    normalization?: EmbeddingNormalization | null;
    signal?: AbortSignal;
  }): Promise<EmbeddingResult>;
  public async embed(options: {
    input: EmbeddingInput;
    normalization?: EmbeddingNormalization | null;
    signal?: AbortSignal;
  }): Promise<EmbeddingResult | SingleEmbeddingResult> {
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

      const response = await requestJson<ApiEmbeddingResponse>({
        fetchFn: this.client.clientOptions.fetchFn,
        baseUrl: this.client.BASE_URL,
        method: "POST",
        pathName: "/embedding",
        body,
        signal: options.signal,
      });

      const embeddings = response.map((item) =>
        Array.isArray(item.embedding[0]) ? item.embedding[0] : item.embedding,
      );

      if (typeof options.input === "string") {
        return embeddings[0]!;
      }

      return embeddings;
    } catch (error) {
      if (error instanceof DOMException && error.code === error.ABORT_ERR) {
        throw new AbortedRequestError("The embedding request was cancelled before completion");
      } else {
        throw new UnexpectedServerBehaviorError(
          "An unexpected error occurred during embedding generation",
          error,
        );
      }
    }
  }
}
