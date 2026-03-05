import type { ChalkInstance } from "chalk";

// these are dynamically imported only if we're in Node
let utilInspect:
  | ((
      object: unknown,
      showHidden?: boolean,
      depth?: number | null,
      color?: boolean
    ) => string)
  | null = null;
let isatty: ((fd: number) => boolean) | null = null;
let chalk: ChalkInstance | null = null;

const isNode =
  typeof process !== "undefined" &&
  typeof process.versions === "object" &&
  !!process.versions.node;

if (isNode) {
  // Lazy-load Node-specific modules
  const nodeUtil = await import("node:util");
  const nodeTty = await import("node:tty");
  utilInspect = nodeUtil.inspect;
  isatty = nodeTty.isatty;
  chalk = (await import(/* @vite-ignore */ "chalk")).default;
}

export class Logger {
  tag: string;
  enabled: boolean;
  sink: NodeJS.WriteStream | null;

  constructor(tag: string, sink?: NodeJS.WriteStream) {
    this.tag = tag;
    this.enabled = true;
    this.sink = isNode ? sink ?? process.stderr : null;
  }

  setEnabled(enabled: boolean) {
    this.enabled = enabled;
    return this;
  }

  #getCallerInfo(): string | null {
    if (!isNode) return null;
    const stack = new Error().stack ?? "";
    const stackLines = stack.split("\n");
    const callerLine = stackLines[4]?.trim();
    const match = callerLine?.match(
      /at ([#a-zA-Z0-9_\- <>\.]+) \((.*?):(\d+):\d+\)/
    );
    if (match) {
      let file = match[2]?.replace("file:///", "").replace("file://", "");
      if (!file) return null;

      file = file.replace("system/Server/src", "Server/src");
      file = file.replace("system/Server/Common/src", "Common/src");

      const line = parseInt(match[3] ?? "-1", 10);
      return `${file}:${line}`;
    }
    return null;
  }

  #objectInspect(object: unknown): string {
    if (typeof object === "string") return object;

    if (utilInspect && isatty) {
      const hasColor = isatty(process.stderr.fd);
      return utilInspect(object, false, null, hasColor);
    }

    try {
      return JSON.stringify(object, null, 2);
    } catch {
      return String(object);
    }
  }

  #colorize(level: "debug" | "info" | "warn" | "error", text: string): string {
    if (chalk) {
      const colors: Record<typeof level, ChalkInstance> = {
        debug: chalk.green,
        info: chalk.blue,
        warn: chalk.yellow,
        error: chalk.red,
      };
      return colors[level](text);
    } else {
      // for browsers: return plain text; color handled in console methods
      return text;
    }
  }

  #print(level: "debug" | "info" | "warn" | "error", ...args: unknown[]) {
    if (!this.enabled) return;

    const date = new Date().toISOString();
    const origin = this.#getCallerInfo() ?? "unknown";
    const prefix = `${date} ${level} [${origin}] ${this.tag}`;

    if (isNode && this.sink) {
      this.sink.write(this.#colorize(level, prefix) + " ");
      this.sink.write(
        args.map(this.#objectInspect.bind(this)).join(" ") + "\n"
      );
    } else {
      // browser fallback: use console methods with color styles
      const styles: Record<typeof level, string> = {
        debug: "color: green;",
        info: "color: blue;",
        warn: "color: goldenrod;",
        error: "color: red;",
      };
      const consoleFn =
        level === "debug"
          ? console.debug
          : level === "info"
          ? console.info
          : level === "warn"
          ? console.warn
          : console.error;

      consoleFn(`%c${prefix}`, styles[level], ...args);
    }
  }

  debug(...args: unknown[]) {
    this.#print("debug", ...args);
  }
  info(...args: unknown[]) {
    this.#print("info", ...args);
  }
  warn(...args: unknown[]) {
    this.#print("warn", ...args);
  }
  error(...args: unknown[]) {
    this.#print("error", ...args);
  }

  local(childName: string) {
    return new Logger(
      `${this.tag}/${childName}`,
      this.sink ?? undefined
    ).setEnabled(this.enabled);
  }
}

export default Logger;
