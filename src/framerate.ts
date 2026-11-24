/** Assumed when the video's frame rate is not known yet (paused, no `requestVideoFrameCallback`). */
export const FALLBACK_FPS = 30;

/** Auto's frame budget never exceeds this, however slow the video: decode and compositing need room too. */
export const MAX_FRAME_BUDGET_MS = 12;
/** Share of the frame interval auto may spend on the GPU. */
export const FRAME_BUDGET_SHARE = 0.35;

/** Rates video actually comes in. Measured intervals snap to the nearest one. */
const COMMON_RATES = [23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60, 90, 100, 119.88, 120];

function validFps(fps: number | null | undefined): number {
  return typeof fps === 'number' && Number.isFinite(fps) && fps > 0 ? fps : FALLBACK_FPS;
}

/**
 * Auto's GPU budget per frame for a video of `fps` frames per second:
 * `min(12 ms, 35% of the frame interval)`. 24 fps film gets 12 ms, 30 fps 11.7 ms, 60 fps 5.8 ms.
 */
export function frameBudgetFor(fps: number | null | undefined): number {
  const interval = 1000 / validFps(fps);
  return Math.min(MAX_FRAME_BUDGET_MS, FRAME_BUDGET_SHARE * interval);
}

/**
 * GPU time above which a frame counts against the running preset: 60% of the frame interval,
 * within 16..33 ms. 60 fps gets 16 ms, 30 fps 20 ms, 24 fps 25 ms.
 */
export function slowFrameFor(fps: number | null | undefined): number {
  const interval = 1000 / validFps(fps);
  return Math.min(33, Math.max(16, interval * 0.6));
}

/** The common rate within 3% of `fps`, or `fps` itself rounded to 0.01. */
export function snapFrameRate(fps: number): number {
  let best: number | null = null;
  let bestError = 0.03;
  for (const rate of COMMON_RATES) {
    const error = Math.abs(fps - rate) / rate;
    if (error < bestError) {
      best = rate;
      bestError = error;
    }
  }
  return best ?? Math.round(fps * 100) / 100;
}

/**
 * The video's frame rate from the media timestamps `requestVideoFrameCallback` reports. The
 * median of the last intervals ignores a dropped frame (a double interval) or a seek.
 */
export class FrameRateEstimator {
  private intervals: number[] = [];
  private last: number | null = null;

  constructor(
    private readonly size = 24,
    private readonly minSamples = 6,
  ) {}

  /** The `mediaTime` (seconds) of a presented frame. */
  push(mediaTime: number): void {
    if (!Number.isFinite(mediaTime)) return;
    const last = this.last;
    this.last = mediaTime;
    if (last === null) return;
    const delta = mediaTime - last;
    // Seeks, loops and stalls: not a frame interval. 4..250 fps covers every real video.
    if (!(delta > 0.004 && delta < 0.25)) return;
    this.intervals.push(delta);
    if (this.intervals.length > this.size) this.intervals.shift();
  }

  /** Frames per second, snapped to a common rate; null until enough frames are in. */
  value(): number | null {
    if (this.intervals.length < this.minSamples) return null;
    const sorted = [...this.intervals].sort((a, b) => a - b);
    const mid = sorted.length >> 1;
    const interval = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    return snapFrameRate(1 / interval);
  }

  reset(): void {
    this.intervals = [];
    this.last = null;
  }
}
