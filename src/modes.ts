/** Everything the user can pick. `auto` resolves to one of the other four at runtime. */
export type Mode = 'auto' | 'off' | 'performance' | 'balanced' | 'quality' | 'ultra';

/** What is actually rendering. `auto` never appears here. */
export type ActiveMode = Exclude<Mode, 'auto'>;

/** A preset that runs shaders, weakest first. */
export type Preset = Exclude<ActiveMode, 'off'>;

/** A preset that needs trained network weights (see `models` in the options). */
export type ModelPreset = Exclude<Preset, 'performance'>;

export const MODES: readonly Mode[] = [
  'auto',
  'off',
  'performance',
  'balanced',
  'quality',
  'ultra',
];

/** Weakest to strongest. */
export const PRESETS: readonly Preset[] = ['performance', 'balanced', 'quality', 'ultra'];

/** The presets that run a luma CNN and only exist when their weights were supplied. */
export const MODEL_PRESETS: readonly ModelPreset[] = ['balanced', 'quality', 'ultra'];

export function isMode(value: unknown): value is Mode {
  return typeof value === 'string' && (MODES as readonly string[]).includes(value);
}

export function isPreset(value: unknown): value is Preset {
  return typeof value === 'string' && (PRESETS as readonly string[]).includes(value);
}

export function isModelPreset(value: unknown): value is ModelPreset {
  return typeof value === 'string' && (MODEL_PRESETS as readonly string[]).includes(value);
}

/**
 * The modes the settings menu offers: `auto`, `off` and every preset that can actually run.
 * `performance` needs nothing but WebGPU, so it is always there.
 */
export function menuModes(available: ReadonlySet<Preset>): Mode[] {
  return MODES.filter((mode) => !isPreset(mode) || mode === 'performance' || available.has(mode));
}

/**
 * The preset that runs for a requested one: itself when available, otherwise the strongest
 * available preset below it, and `performance` at the very least. A `balanced` or `quality` pick
 * without weights therefore renders `performance`.
 */
export function effectivePreset(requested: Preset, available: ReadonlySet<Preset>): Preset {
  for (let i = PRESETS.indexOf(requested); i > 0; i--) {
    if (available.has(PRESETS[i])) return PRESETS[i];
  }
  return 'performance';
}

/**
 * One step weaker, skipping presets that are not available. `performance` steps down to `off`,
 * and `off` stays `off`.
 */
export function stepDown(
  mode: ActiveMode,
  available: ReadonlySet<Preset> = new Set(PRESETS),
): ActiveMode {
  if (mode === 'off') return 'off';
  for (let i = PRESETS.indexOf(mode) - 1; i >= 0; i--) {
    if (PRESETS[i] === 'performance' || available.has(PRESETS[i])) return PRESETS[i];
  }
  return 'off';
}

/** Median per-frame time of one preset, measured on this device at this video's resolution. */
export interface BenchmarkSample {
  preset: Preset;
  /** Median milliseconds from submit to GPU completion. `Infinity` when the preset failed to build. */
  ms: number;
}

/**
 * The strongest preset whose median frame time fits the budget; `off` when none does. Only presets
 * that were measured count, so auto can never pick a preset whose weights are missing.
 */
export function pickAutoMode(samples: readonly BenchmarkSample[], budgetMs: number): ActiveMode {
  let best: ActiveMode = 'off';
  for (const preset of PRESETS) {
    const sample = samples.find((s) => s.preset === preset);
    if (sample && Number.isFinite(sample.ms) && sample.ms <= budgetMs) best = preset;
  }
  return best;
}

/** Median of a list of numbers; `Infinity` for an empty list so it can never pass a budget. */
export function median(values: readonly number[]): number {
  if (values.length === 0) return Infinity;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}
