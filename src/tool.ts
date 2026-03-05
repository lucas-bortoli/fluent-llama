import * as v from "valibot";

/**
 * Type alias for a generic Valibot schema.
 */
export type ValibotSchema = v.BaseSchema<unknown, unknown, v.BaseIssue<unknown>>;

/**
 * Defines the structure of a tool callable by an LLM.
 *
 * @template N The name of the tool (string literal).
 * @template P The parameters schema (Valibot object).
 */
export interface Tool<N extends string, P extends Record<string, ValibotSchema>> {
  /** The unique name of the tool. */
  name: N;
  /** Description provided to the model. */
  description: string;
  /** Valibot schema for parameter validation. */
  parameters: P;
  /** The execution logic for the tool. */
  exec: (parameters: { [K in keyof P]: v.InferOutput<P[K]> }) => Promise<object>;
}

/**
 * Factory function to create a new `Tool` instance.
 *
 * @param opts Configuration options for the tool.
 * @param opts.name The name of the tool.
 * @param opts.description A description for the model.
 * @param opts.parameters Valibot schema for arguments.
 * @param opts.exec The function to execute when the tool is called.
 * @returns A configured `Tool` instance.
 */
export function tool<N extends string, P extends Record<string, ValibotSchema>>(opts: {
  name: N;
  description: string;
  parameters: P;
  exec: (parameters: { [K in keyof P]: v.InferOutput<P[K]> }) => Promise<object>;
}): Tool<N, P> {
  return opts;
}

/**
 * Defines how a toolset should invoke tools.
 */
export type InvocationRequirement<N extends string> =
  | "AsNeeded"
  | "RequireOne"
  | { mode: "RequireOneSpecific"; tool: N };

/**
 * Defines the execution strategy for multiple tools.
 */
export type BatchMode = "Sequential" | "Parallel";

/**
 * Callback handler for when a tool call starts.
 * @param tool The tool that is starting.
 */
export type CallbackOnToolCallStart<T extends Tool<string, any>> = (tool: T) => void;

/**
 * Callback handler for when a tool call ends.
 * @param tool The tool that has finished.
 */
export type CallbackOnToolCallEnd<T extends Tool<string, any>> = (tool: T) => void;

/**
 * Manages a collection of tools and configuration settings.
 *
 * Uses a fluent API for configuration.
 */
export class Toolset<T extends Tool<string, any>, N extends string = T["name"]> {
  /** Map of tool names to tool instances. */
  public readonly tools: Map<string, T>;

  private invocationRequirement: InvocationRequirement<N>;
  private batchMode: BatchMode;
  private whitelist: N[];
  private callbackOnToolCallStart: CallbackOnToolCallStart<T> | null;
  private callbackOnToolCallEnd: CallbackOnToolCallEnd<T> | null;

  /**
   * Creates a new Toolset instance.
   * @param tools An array of tools to include in the set.
   */
  public constructor(tools: T[]) {
    this.tools = new Map(tools.map((t) => [t.name, t]));
    this.invocationRequirement = "AsNeeded";
    this.batchMode = "Parallel";
    this.whitelist = tools.map((t) => t.name as N);
    this.callbackOnToolCallStart = null;
    this.callbackOnToolCallEnd = null;
  }

  /**
   * Sets the requirement for tool invocation.
   * @param requirement The invocation strategy.
   * @returns This Toolset instance for chaining.
   */
  public setInvocationRequirement(requirement: InvocationRequirement<N>) {
    return this;
  }

  /**
   * Filters the available tools based on a whitelist.
   * @param whitelisted List of tool names allowed to run.
   * @returns This Toolset instance for chaining.
   */
  public setWhitelist(whitelisted: N[]) {
    this.whitelist = whitelisted.filter((name) => this.tools.has(name));
    return this;
  }

  /**
   * Sets whether tools should run sequentially or in parallel.
   * @param mode The execution mode.
   * @returns This Toolset instance for chaining.
   */
  public setBatchMode(mode: BatchMode) {
    this.batchMode = mode;
    return this;
  }

  /**
   * Sets a handler to call when a tool call begins.
   * @param handler The callback function.
   * @returns This Toolset instance for chaining.
   */
  public setCallbackOnToolCallStart(handler: CallbackOnToolCallStart<T> | null) {
    this.callbackOnToolCallStart = handler;
    return this;
  }

  /**
   * Sets a handler to call when a tool call ends.
   * @param handler The callback function.
   * @returns This Toolset instance for chaining.
   */
  public setCallbackOnToolCallEnd(handler: CallbackOnToolCallEnd<T> | null) {
    this.callbackOnToolCallEnd = handler;
    return this;
  }

  /**
   * Helper for Debugging: Long chains can be harder to set breakpoints on.
   * This function immediately calls its predicate in a chain.
   * @param lambda The function to execute immediately.
   * @returns This Toolset instance for chaining.
   */
  public __(lambda: (sampling: typeof this) => void) {
    lambda(this);
    return this;
  }

  /**
   * Compiles the configuration into an immutable `ToolsetResult`.
   * @returns The finalized toolset configuration.
   */
  public build(): ToolsetResult {
    return {
      tools: new Map(
        this.tools.values().map((tool) => {
          const toolResult = {
            name: tool.name,
            description: tool.description,
            parameters: tool.parameters,
            parametersSchema: v.object(tool.parameters),
            exec: tool.exec,
          } satisfies ToolResult;

          return [toolResult.name, toolResult];
        }),
      ),
      whitelist: this.whitelist,
      invocationRequirement: this.invocationRequirement,
      batchMode: this.batchMode,
      callbacks: {
        onStart: this.callbackOnToolCallStart as CallbackOnToolCallStart<ToolResult> | null,
        onEnd: this.callbackOnToolCallEnd as CallbackOnToolCallEnd<ToolResult> | null,
      },
    };
  }
}

/**
 * Represents a compiled tool ready for runtime execution.
 */
export interface ToolResult {
  name: string;
  description: string;
  parameters: Record<string, ValibotSchema>;
  parametersSchema: ValibotSchema;
  exec: (parameters: Record<string, object>) => Promise<object>;
}

/**
 * Represents the finalized configuration of a Toolset.
 */
export interface ToolsetResult {
  readonly tools: ReadonlyMap<string, ToolResult>;
  readonly whitelist: ReadonlyArray<string>;
  readonly invocationRequirement: InvocationRequirement<string>;
  readonly batchMode: BatchMode;
  readonly callbacks: {
    readonly onStart: CallbackOnToolCallStart<ToolResult> | null;
    readonly onEnd: CallbackOnToolCallEnd<ToolResult> | null;
  };
}
