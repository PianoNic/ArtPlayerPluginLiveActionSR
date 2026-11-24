import type { Size } from './layout.js';
import type { Preset } from './modes.js';

/** Strengths of the hand-tuned stages, each 0..1; 0 drops the stage. */
export interface Tuning {
  /** Deband strength at source resolution. */
  deband: number;
  /** RCAS sharpening at output resolution. */
  sharpness: number;
  /** Luma film grain added at output resolution. */
  grain: number;
}

export const DEFAULT_TUNING: Tuning = { deband: 0.5, sharpness: 0.4, grain: 0.25 };

/** The luma CNN only runs when the screen is this much larger than the video on both axes. */
export const MODEL_MIN_RATIO = 1.2;
/** EASU only runs when the target is this much larger on at least one axis. */
export const EASU_MIN_RATIO = 1.05;

export type Stage =
  /** f3kdb-style deband, at the source resolution. */
  | { kind: 'deband'; size: Size }
  /** The CNN at `scale`x: a luma residual on a bilinear upscale, or (graph models) the RGB frame. */
  | { kind: 'model'; from: Size; size: Size; scale: number }
  /** FSR 1 EASU, edge-adaptive upscale. */
  | { kind: 'easu'; from: Size; size: Size }
  /** FSR 1 RCAS, sharpening. */
  | { kind: 'rcas'; size: Size };

export interface PassPlan {
  stages: Stage[];
  /** Size of the last stage's output; the present pass scales it to the canvas. */
  output: Size;
  /** Whether any stage raises the resolution. */
  upscaling: boolean;
  /** Whether the luma CNN runs. */
  model: boolean;
}

function larger(target: Size, base: Size, ratio: number, bothAxes: boolean): boolean {
  const x = target.width > base.width * ratio;
  const y = target.height > base.height * ratio;
  return bothAxes ? x && y : x || y;
}

function notSmaller(target: Size, base: Size): boolean {
  return target.width >= base.width && target.height >= base.height;
}

/**
 * The stages a preset runs for a video of `native` size shown at `target` device pixels:
 *
 *   performance  deband > EASU to target > RCAS               (> grain when presenting)
 *   balanced /   deband > CNN x2 > EASU to target if the CNN fell short > RCAS
 *   quality /
 *   ultra
 *
 * When the screen is not larger than the video, nothing is upscaled: the frame gets the 1x
 * cleanup (deband, RCAS, grain) and the present pass scales it. The CNN needs weights
 * (`modelScale`) and a screen more than `MODEL_MIN_RATIO` the size of the video; without either,
 * a model preset runs the `performance` stages.
 */
export function planPasses(
  preset: Preset,
  native: Size,
  target: Size,
  tuning: Tuning,
  modelScale: number | null,
): PassPlan {
  const stages: Stage[] = [];
  let size: Size = { ...native };
  let model = false;

  if (tuning.deband > 0) stages.push({ kind: 'deband', size });

  if (preset !== 'performance' && modelScale && larger(target, native, MODEL_MIN_RATIO, true)) {
    const next = { width: native.width * modelScale, height: native.height * modelScale };
    stages.push({ kind: 'model', from: size, size: next, scale: modelScale });
    size = next;
    model = true;
  }

  if (notSmaller(target, size) && larger(target, size, EASU_MIN_RATIO, false)) {
    const next = { ...target };
    stages.push({ kind: 'easu', from: size, size: next });
    size = next;
  }

  if (tuning.sharpness > 0) stages.push({ kind: 'rcas', size });

  return {
    stages,
    output: size,
    upscaling: size.width > native.width || size.height > native.height,
    model,
  };
}

/** Deband threshold, in 0..1 signal units: strength 0.5 is 3/255. */
export function debandThreshold(strength: number): number {
  return (Math.min(1, Math.max(0, strength)) * 6) / 255;
}

/** Deband search radius in source pixels: 16 at 1080p, scaled with the video height, at least 4. */
export function debandRadius(native: Size): number {
  return Math.max(4, Math.round((16 * native.height) / 1080));
}

/**
 * RCAS's sharpness constant: the linear multiplier on its negative lobe. AMD expresses it in
 * stops (`2^-stops`); strength 1 is 0 stops (full), 0.5 is 1 stop, 0.4 (the default) 1.2 stops.
 */
export function rcasSharpness(strength: number): number {
  const s = Math.min(1, Math.max(0, strength));
  return s === 0 ? 0 : 2 ** -(2 * (1 - s));
}

/** Peak amplitude of the luma grain in 0..1 signal units: strength 0.25 is about 2.5/255. */
export function grainAmplitude(strength: number): number {
  return Math.min(1, Math.max(0, strength)) * 0.04;
}
