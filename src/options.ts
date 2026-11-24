import type { ModelSource } from './model.js';
import { isMode, MODEL_PRESETS, type ActiveMode, type Mode, type ModelPreset } from './modes.js';
import { DEFAULT_TUNING, type Tuning } from './plan.js';

/** Every user-facing string. Pass your own to translate the setting entry. */
export interface LiveActionSrLabels {
  /** Name of the entry in the settings menu. */
  setting: string;
  auto: string;
  off: string;
  performance: string;
  balanced: string;
  quality: string;
  ultra: string;
  /** Tooltip while auto is selected. `{mode}` is replaced with the label of the preset auto chose. */
  autoActive: string;
  /** Tooltip when the browser cannot run WebGPU. */
  unsupported: string;
  /** Notice, shown once per source, when a preset the viewer picked cannot keep up. */
  struggling: string;
  /** Notice when the video is cross-origin without CORS and its frames cannot be read. */
  blocked: string;
}

export const DEFAULT_LABELS: LiveActionSrLabels = {
  setting: 'Enhance',
  auto: 'Auto',
  off: 'Off',
  performance: 'Performance',
  balanced: 'Balanced',
  quality: 'Quality',
  ultra: 'Ultra',
  autoActive: 'Auto ({mode})',
  unsupported: 'Not supported',
  struggling: 'Enhancing is heavy for this GPU; try a lighter preset',
  blocked: 'This video cannot be enhanced: its server does not allow it (CORS)',
};

export interface LiveActionSrOptions {
  /** Starting mode. Default `auto`. */
  mode?: Mode;
  /** Replaces any of the default English labels. */
  labels?: Partial<LiveActionSrLabels>;
  /**
   * Weights for the CNN presets: an object in the documented JSON format, a URL to fetch it from,
   * or a function that resolves to it. Loaded on first use. A preset without weights is not
   * offered, auto never picks it, and `setMode()` with it renders `performance`.
   */
  models?: Partial<Record<ModelPreset, ModelSource>>;
  /** Strength of deband, sharpening and grain, each 0..1; 0 turns a stage off. */
  tuning?: Partial<Tuning>;
  /**
   * Fires when the selected mode or the preset actually rendering changes - a pick in the menu,
   * `setMode()`, auto settling after its benchmark, or a runtime downgrade.
   */
  onModeChange?: (mode: Mode, active: ActiveMode) => void;
  /** Fires when enhancing had to stop: no WebGPU, a cross-origin video, a lost GPU device. */
  onError?: (error: unknown) => void;
  /** Add an entry to ArtPlayer's settings menu. Default `true`. Needs `setting: true` on the player. */
  setting?: boolean;
  /** SVG or HTML for the settings entry icon. */
  icon?: string;
  /** Split view: the left half shows the original video, the right half the enhanced one. */
  compare?: boolean;
  /**
   * Auto picks the strongest preset whose median GPU time per frame stays under this. Default:
   * derived from the video's frame rate, `min(12 ms, 35% of the frame interval)`.
   */
  frameBudgetMs?: number;
  /**
   * Frames taking longer than this on the GPU count against the current preset. Default: derived
   * from the frame rate, 60% of the frame interval within 16..33 ms.
   */
  slowFrameMs?: number;
  /**
   * Let auto step down one preset when the GPU cannot keep up. Default `true`. A preset the viewer
   * picked is never lowered either way; they get a one-time notice (`labels.struggling`) instead.
   */
  autoDowngrade?: boolean;
  /**
   * Delay the video's audio by as much as the enhanced picture trails the plain video, so sound and
   * picture stay in step. Routes the element's audio through Web Audio. Default `true`.
   */
  syncAudio?: boolean;
  /** Upper bound on the canvas backing store, in pixels. Default 3840 x 2160. */
  maxOutputPixels?: number;
  /** Remember the auto benchmark per device in localStorage. Default `true`. */
  cache?: boolean;
  /** localStorage key prefix. Default `artplayer-plugin-liveaction-sr`. */
  cacheKey?: string;
  /** Log decisions (benchmark numbers, downgrades) to the console. Default `false`. */
  debug?: boolean;
}

export interface ResolvedOptions {
  mode: Mode;
  labels: LiveActionSrLabels;
  models: Partial<Record<ModelPreset, ModelSource>>;
  tuning: Tuning;
  onModeChange: ((mode: Mode, active: ActiveMode) => void) | null;
  onError: ((error: unknown) => void) | null;
  setting: boolean;
  icon: string;
  compare: boolean;
  /** null: derived from the frame rate. */
  frameBudgetMs: number | null;
  /** null: derived from the frame rate. */
  slowFrameMs: number | null;
  autoDowngrade: boolean;
  syncAudio: boolean;
  maxOutputPixels: number;
  cache: boolean;
  cacheKey: string;
  debug: boolean;
}

/**
 * Lucide "film" (ISC licence). `fill:none` is inline because ArtPlayer fills every svg in the
 * player, which would turn the outline into a solid blob.
 */
export const DEFAULT_ICON =
  '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" style="fill:none" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M7 3v18"/><path d="M3 7.5h4"/><path d="M3 12h18"/><path d="M3 16.5h4"/><path d="M17 3v18"/><path d="M17 7.5h4"/><path d="M17 16.5h4"/></svg>';

export const DEFAULT_CACHE_KEY = 'artplayer-plugin-liveaction-sr';

function positive(value: unknown, fallback: number, min: number, max: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) return fallback;
  return Math.min(max, Math.max(min, value));
}

function optionalPositive(value: unknown, min: number, max: number): number | null {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) return null;
  return Math.min(max, Math.max(min, value));
}

function unit(value: unknown, fallback: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return fallback;
  return Math.min(1, Math.max(0, value));
}

function nonEmptyString(value: unknown, fallback: string): string {
  return typeof value === 'string' && value.trim() !== '' ? value : fallback;
}

function isModelSource(value: unknown): value is ModelSource {
  return (
    (typeof value === 'string' && value.trim() !== '') ||
    typeof value === 'function' ||
    (typeof URL !== 'undefined' && value instanceof URL) ||
    (!!value && typeof value === 'object')
  );
}

/** Fills in defaults and drops invalid values. Never throws. */
export function resolveOptions(
  input: LiveActionSrOptions | null | undefined = {},
): ResolvedOptions {
  const o = (input && typeof input === 'object' ? input : {}) as LiveActionSrOptions;
  const labels = { ...DEFAULT_LABELS };
  if (o.labels && typeof o.labels === 'object') {
    for (const key of Object.keys(DEFAULT_LABELS) as (keyof LiveActionSrLabels)[]) {
      labels[key] = nonEmptyString(o.labels[key], DEFAULT_LABELS[key]);
    }
  }
  const models: Partial<Record<ModelPreset, ModelSource>> = {};
  if (o.models && typeof o.models === 'object') {
    for (const preset of MODEL_PRESETS) {
      if (isModelSource(o.models[preset])) models[preset] = o.models[preset];
    }
  }
  const tuning = (o.tuning && typeof o.tuning === 'object' ? o.tuning : {}) as Partial<Tuning>;
  return {
    mode: isMode(o.mode) ? o.mode : 'auto',
    labels,
    models,
    tuning: {
      deband: unit(tuning.deband, DEFAULT_TUNING.deband),
      sharpness: unit(tuning.sharpness, DEFAULT_TUNING.sharpness),
      grain: unit(tuning.grain, DEFAULT_TUNING.grain),
    },
    onModeChange: typeof o.onModeChange === 'function' ? o.onModeChange : null,
    onError: typeof o.onError === 'function' ? o.onError : null,
    setting: o.setting !== false,
    icon: nonEmptyString(o.icon, DEFAULT_ICON),
    compare: o.compare === true,
    frameBudgetMs: optionalPositive(o.frameBudgetMs, 1, 100),
    slowFrameMs: optionalPositive(o.slowFrameMs, 1, 200),
    autoDowngrade: o.autoDowngrade !== false,
    syncAudio: o.syncAudio !== false,
    maxOutputPixels: Math.round(positive(o.maxOutputPixels, 3840 * 2160, 320 * 180, 7680 * 4320)),
    cache: o.cache !== false,
    cacheKey: nonEmptyString(o.cacheKey, DEFAULT_CACHE_KEY),
    debug: o.debug === true,
  };
}

/** The label shown next to the settings entry for the current state. */
export function describeMode(
  mode: Mode,
  active: ActiveMode,
  labels: LiveActionSrLabels,
  supported: boolean,
): string {
  if (!supported) return labels.unsupported;
  if (mode === 'auto') return labels.autoActive.replace('{mode}', labels[active]);
  return labels[mode];
}
