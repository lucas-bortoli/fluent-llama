/**
 * Symbol used to disable a specific sampler instance.
 */
export const Disabled = Symbol("SamplerDisabled");
export type Disabled = typeof Disabled;

/**
 * Symbol used to specify a random seed instead of a fixed one.
 */
export const RandomSeed = Symbol("RandomSeed");
export type RandomSeed = typeof RandomSeed;

/**
 * Configuration interface for the temperature parameter.
 */
interface TemperatureConfig {
  value: number;
}

/**
 * Configuration interface for Top-K sampling.
 */
interface TopKConfig {
  value: number;
}

/**
 * Configuration interface for Top-P (Nucleus) sampling.
 */
interface TopPConfig {
  value: number;
}

/**
 * Configuration interface for Min-P sampling.
 */
interface MinPConfig {
  value: number;
}

/**
 * Configuration interface for Top-N-Sigma sampling.
 */
interface TopNSigmaConfig {
  value: number;
}

/**
 * Configuration interface for XTC (Exclude Top Choices) sampling.
 */
interface XtcConfig {
  probability: number;
  threshold: number;
}

/**
 * Configuration interface for Repetition penalty.
 */
interface RepetitionConfig {
  lastN: number;
  penalty: number;
}

/**
 * Configuration interface for Locally Typical sampling.
 */
interface TypicalConfig {
  value: number;
}

/**
 * Configuration interface for Mirostat sampling.
 */
interface MirostatConfig {
  version: 0 | 1 | 2;
  lr: number;
  ent: number;
}

/**
 * Configuration interface for DRY (Don't Repeat Yourself) sampling.
 */
interface DryConfig {
  multiplier: number;
  base: number;
  allowedLength: number;
  penaltyLastN: number;
  sequenceBreaker: string;
}

/**
 * Configuration interface for Adaptive-P sampling.
 */
interface AdaptiveConfig {
  target: number;
  decay: number;
}

/**
 * Configuration interface for Dynamic Temperature.
 */
interface DynatempConfig {
  range: number;
  exp: number;
}

/**
 * Configuration interface for Presence Penalty.
 */
interface PresencePenaltyConfig {
  value: number;
}

/**
 * Configuration interface for Frequency Penalty.
 */
interface FrequencyPenaltyConfig {
  value: number;
}

const VALID_SAMPLER_NAMES = [
  "penalties",
  "dry",
  "top_n_sigma",
  "top_k",
  "typ_p",
  "top_p",
  "min_p",
  "xtc",
  "temperature",
] as const;

export type SamplerName = (typeof VALID_SAMPLER_NAMES)[number];

export const BanToken = Symbol.for("Sampling/BanToken");
export type BanToken = typeof BanToken;

export type Grammar = { type: "Json"; schema: object } | { type: "Gbnf"; grammar: string };

export class Sampling {
  private static readonly defaultConfig: SamplingState = {
    temperature: { value: 0.8 },
    top_k: { value: 40 },
    top_p: { value: 0.95 },
    min_p: { value: 0.05 },
    top_n_sigma: { value: -1 },
    xtc: { probability: 0, threshold: 0 },
    repetition: { lastN: 64, penalty: 1 },
    typical: { value: 1 },
    mirostat: { version: 0, lr: 0.1, ent: 5 },
    dry: { multiplier: 0, base: 1.75, allowedLength: 2, penaltyLastN: -1, sequenceBreaker: "" },
    adaptive: { target: -1, decay: 0.9 },
    dynatemp: { range: 0, exp: 1 },
    presence_penalty: 0,
    frequency_penalty: 0,
    seed: -1,
  };

  private static readonly defaultOrder: SamplerName[] = [
    "penalties",
    "dry",
    "top_n_sigma",
    "top_k",
    "typ_p",
    "top_p",
    "min_p",
    "xtc",
    "temperature",
  ];

  private config: SamplingState = { ...Sampling.defaultConfig };
  private order: SamplerName[] = [...Sampling.defaultOrder];
  private readonly bias: Map<number, number | BanToken> = new Map();
  private grammar: Grammar | null = null;

  /**
   * Sets the temperature for generation.
   */
  public setSamplerTemperature(value: number | Disabled): this {
    this.config.temperature = value === Disabled ? undefined : { value };
    return this;
  }

  /**
   * Sets the Top-K sampling limit.
   */
  public setSamplerTopK(value: number | Disabled): this {
    this.config.top_k = value === Disabled ? undefined : { value };
    return this;
  }

  /**
   * Sets the Top-P (Nucleus) sampling threshold.
   */
  public setSamplerTopP(value: number | Disabled): this {
    this.config.top_p = value === Disabled ? undefined : { value };
    return this;
  }

  /**
   * Sets the Min-P sampling threshold.
   */
  public setSamplerMinP(value: number | Disabled): this {
    this.config.min_p = value === Disabled ? undefined : { value };
    return this;
  }

  /**
   * Sets the Top-N-Sigma sampling threshold.
   */
  public setSamplerTopNSigma(value: number | Disabled): this {
    this.config.top_n_sigma = value === Disabled ? undefined : { value };
    return this;
  }

  /**
   * Sets XTC sampling parameters.
   */
  public setSamplerXtc(config: XtcConfig | Disabled): this {
    this.config.xtc = config === Disabled ? undefined : config;
    return this;
  }

  /**
   * Sets Repetition penalty parameters.
   */
  public setSamplerRepetitionPenalty(config: RepetitionConfig | Disabled): this {
    this.config.repetition = config === Disabled ? undefined : config;
    return this;
  }

  /**
   * Sets Locally Typical sampling threshold.
   */
  public setSamplerTypical(value: number | Disabled): this {
    this.config.typical = value === Disabled ? undefined : { value };
    return this;
  }

  /**
   * Sets Mirostat sampling parameters.
   */
  public setSamplerMirostat(config: MirostatConfig | Disabled): this {
    this.config.mirostat = config === Disabled ? undefined : config;
    return this;
  }

  /**
   * Sets DRY sampling parameters.
   */
  public setSamplerDry(config: DryConfig | Disabled): this {
    this.config.dry = config === Disabled ? undefined : config;
    return this;
  }

  /**
   * Sets Adaptive-P sampling parameters.
   */
  public setSamplerAdaptive(config: AdaptiveConfig | Disabled): this {
    this.config.adaptive = config === Disabled ? undefined : config;
    return this;
  }

  /**
   * Sets Dynamic Temperature parameters.
   */
  public setSamplerDynatemp(config: DynatempConfig | Disabled): this {
    this.config.dynatemp = config === Disabled ? undefined : config;
    return this;
  }

  /**
   * Sets Presence Penalty parameter.
   * Affects how much to penalize new tokens based on their existing frequency in the text.
   */
  public setSamplerPresencePenalty(value: number | Disabled): this {
    this.config.presence_penalty = value === Disabled ? 0 : value;
    return this;
  }

  /**
   * Sets Frequency Penalty parameter.
   * Affects how much to penalize new tokens based on their existing frequency in the text.
   */
  public setSamplerFrequencyPenalty(value: number | Disabled): this {
    this.config.frequency_penalty = value === Disabled ? 0 : value;
    return this;
  }

  /**
   * Sets the Random Seed (Random if Symbol) or a specific integer.
   */
  public setSeed(value: number | RandomSeed): this {
    this.config.seed = value;
    return this;
  }

  /**
   * Sets the order of samplers for generation.
   */
  public setOrder(order: SamplerName[]): this {
    const validNames = new Set(VALID_SAMPLER_NAMES);
    if (!order.every((name) => validNames.has(name))) {
      throw new Error(
        `Invalid sampler name in order array. Allowed: ${VALID_SAMPLER_NAMES.join(", ")}`,
      );
    }
    this.order = order;
    return this;
  }

  /**
   * Modify the likelihood of a token appearing in the generated text completion.
   * @example sampling.setTokenBias(15043, 1.0) // increase the likelihood of the token 'Hello'
   * @example sampling.setTokenBias(15043, -1.0) // decrease its likelihood
   * @param token The token to be affected.
   * @param bias The bias factor.
   */
  public setTokenBias(token: number, bias: number | BanToken) {
    this.bias.set(token, bias);
    return this;
  }

  /**
   * Sets a grammar for constrained text generation. A grammar defines the rules for generating valid text based on a specific structure or pattern.
   * You can specify either a JSON schema object, for enforcing a valid JSON output, or a GBNF (GGML BNF) grammar string.
   *
   * @warn **This is incompatible with tool calling.**
   */
  public setGrammar(grammar: Grammar | null) {
    this.grammar = grammar;
    return this;
  }

  /**
   * Helper for Debugging: Long chains can be harder to set breakpoints on. This function immediately calls its predicate in a chain.
   */
  public __(lambda: (sampling: typeof this) => void) {
    lambda(this);
    return this;
  }

  /**
   * Builds and returns the final configuration object.
   */
  public build(): SamplingResult {
    const state = this.config;

    return {
      seed: state.seed === RandomSeed ? -1 : state.seed,

      // Sampling Params
      temperature: state.temperature?.value ?? 0.8,
      dynatemp_range: state.dynatemp?.range ?? 0,
      dynatemp_exponent: state.dynatemp?.exp ?? 1,
      top_k: state.top_k?.value ?? 40,
      top_p: state.top_p?.value ?? 0.95,
      min_p: state.min_p?.value ?? 0.05,
      top_nsigma: state.top_n_sigma?.value ?? -1,
      xtc_probability: state.xtc?.probability ?? 0,
      xtc_threshold: state.xtc?.threshold ?? 0,
      typical_p: state.typical?.value ?? 1,
      repeat_last_n: state.repetition?.lastN ?? 64,
      repeat_penalty: state.repetition?.penalty ?? 1,
      presence_penalty: state.presence_penalty ?? 0,
      frequency_penalty: state.frequency_penalty ?? 0,
      dry_multiplier: state.dry?.multiplier ?? 0,
      dry_base: state.dry?.base ?? 1.75,
      dry_allowed_length: state.dry?.allowedLength ?? 2,
      dry_penalty_last_n: state.dry?.penaltyLastN ?? -1,
      dry_sequence_breakers: state.dry?.sequenceBreaker
        ? [state.dry.sequenceBreaker]
        : ["\n", ":", '"', "*"],

      mirostat: state.mirostat?.version ?? 0,
      mirostat_tau: state.mirostat?.ent ?? 5.0,
      mirostat_eta: state.mirostat?.lr ?? 0.1,
      // sampler order
      samplers: this.order,
      logit_bias: Object.fromEntries(
        this.bias.entries().map(([token, bias]) => {
          return [token, bias === BanToken ? false : bias];
        }),
      ),
      grammar: this.grammar?.type === "Gbnf" ? this.grammar.grammar : undefined,
      json_schema: this.grammar?.type === "Json" ? this.grammar.schema : undefined,
    };
  }
}

interface SamplingState {
  temperature: TemperatureConfig | undefined;
  top_k: TopKConfig | undefined;
  top_p: TopPConfig | undefined;
  min_p: MinPConfig | undefined;
  top_n_sigma: TopNSigmaConfig | undefined;
  xtc: XtcConfig | undefined;
  repetition: RepetitionConfig | undefined;
  typical: TypicalConfig | undefined;
  mirostat: MirostatConfig | undefined;
  dry: DryConfig | undefined;
  adaptive: AdaptiveConfig | undefined;
  dynatemp: DynatempConfig | undefined;
  presence_penalty: number | undefined;
  frequency_penalty: number | undefined;
  seed: number | RandomSeed;
}

export type SamplingResult = {
  readonly seed: number;
  readonly temperature: number;
  readonly dynatemp_range: number;
  readonly dynatemp_exponent: number;
  readonly top_k: number;
  readonly top_p: number;
  readonly min_p: number;
  readonly top_nsigma: number;
  readonly xtc_probability: number;
  readonly xtc_threshold: number;
  readonly typical_p: number;
  readonly repeat_last_n: number;
  readonly repeat_penalty: number;
  readonly presence_penalty: number;
  readonly frequency_penalty: number;
  readonly dry_multiplier: number;
  readonly dry_base: number;
  readonly dry_allowed_length: number;
  readonly dry_penalty_last_n: number;
  readonly dry_sequence_breakers: ReadonlyArray<string>;
  readonly mirostat: number;
  readonly mirostat_tau: number;
  readonly mirostat_eta: number;
  readonly samplers: ReadonlyArray<string>;
  readonly logit_bias: { readonly [token: number]: number | false };

  // these are mutually exclusive
  readonly grammar?: string | undefined;
  readonly json_schema?: object | undefined;
};
