import { objectToCamelCase } from "./helpers.js";

/**
 * A generator that parses a Server-Sent Events (SSE) JSON stream and yields parsed chunks.
 * Handles network fragmentation and incomplete lines efficiently.
 *
 * @param stream The ReadableStream of Uint8Array chunks from the API.
 * @yields The parsed and camel-cased JSON objects.
 */
export async function* apiStreamChunker<Chunk extends object>(
  stream: ReadableStream,
): AsyncGenerator<Chunk> {
  // Instantiate decoder once to avoid overhead in loop
  const decoder = new TextDecoder();
  let buffer = "";

  for await (const chunk of stream) {
    // append chunk to buffer (stream: true handles partial multi-byte chars)
    buffer += decoder.decode(chunk, { stream: true });

    // process all complete lines currently in the buffer
    while (true) {
      const endOfLineIndex = buffer.indexOf("\n");

      if (endOfLineIndex === -1) {
        // buffer contains an incomplete line; wait for the next chunk
        break;
      }

      const line = buffer.slice(0, endOfLineIndex).trim();

      // remove the line from the buffer (keep the remainder for next iteration)
      buffer = buffer.slice(endOfLineIndex + 1);

      if (line.length === 0) continue;

      // is this data payload? extract content after "data:"
      if (!line.startsWith("data:")) continue;
      const content = line.slice(5).trim();

      // check for end of stream marker
      if (content === "[DONE]") {
        return;
      }

      try {
        const json = JSON.parse(content);
        yield objectToCamelCase(json, true) as Chunk;
      } catch {
        // ignore malformed chunks to prevent generator crash
        continue;
      }
    }
  }

  // if stream ends without a newline, the remaining buffer is discarded
  // (standard SSE behavior for incomplete trailing data)
}
