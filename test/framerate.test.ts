import { describe, expect, it } from 'vitest';
import {
  FALLBACK_FPS,
  FrameRateEstimator,
  frameBudgetFor,
  MAX_FRAME_BUDGET_MS,
  slowFrameFor,
  snapFrameRate,
} from '../src/framerate';

/** Media timestamps of `n` frames at `fps`, starting at `start` seconds. */
function timestamps(fps: number, n: number, start = 10): number[] {
  return Array.from({ length: n }, (_, i) => start + i / fps);
}

describe('frameBudgetFor', () => {
  it('gives slow film more time than fast TV', () => {
    expect(frameBudgetFor(24)).toBeGreaterThan(frameBudgetFor(60));
    expect(frameBudgetFor(25)).toBeGreaterThan(frameBudgetFor(50));
  });

  it('is 35% of the frame interval, capped at 12 ms', () => {
    expect(frameBudgetFor(24)).toBe(MAX_FRAME_BUDGET_MS);
    expect(frameBudgetFor(30)).toBeCloseTo(11.667, 3);
    expect(frameBudgetFor(60)).toBeCloseTo(5.833, 3);
    expect(frameBudgetFor(120)).toBeCloseTo(2.917, 3);
  });

  it('assumes the fallback rate when the rate is unknown or nonsense', () => {
    const fallback = frameBudgetFor(FALLBACK_FPS);
    expect(frameBudgetFor(null)).toBe(fallback);
    expect(frameBudgetFor(undefined)).toBe(fallback);
    expect(frameBudgetFor(0)).toBe(fallback);
    expect(frameBudgetFor(-24)).toBe(fallback);
    expect(frameBudgetFor(NaN)).toBe(fallback);
  });
});

describe('slowFrameFor', () => {
  it('keeps the classic 16 ms at 60 fps and relaxes for slower video', () => {
    expect(slowFrameFor(60)).toBe(16);
    expect(slowFrameFor(30)).toBeCloseTo(20, 5);
    expect(slowFrameFor(24)).toBeCloseTo(25, 5);
  });

  it('stays within 16..33 ms', () => {
    expect(slowFrameFor(120)).toBe(16);
    expect(slowFrameFor(8)).toBe(33);
  });
});

describe('snapFrameRate', () => {
  it('snaps measured rates to the common ones', () => {
    expect(snapFrameRate(23.9)).toBe(23.976);
    expect(snapFrameRate(24.02)).toBe(24);
    expect(snapFrameRate(29.95)).toBe(29.97);
    expect(snapFrameRate(59.7)).toBe(59.94);
  });

  it('keeps unusual rates, rounded', () => {
    expect(snapFrameRate(15.0004)).toBe(15);
    expect(snapFrameRate(37.123)).toBe(37.12);
  });
});

describe('FrameRateEstimator', () => {
  it('knows nothing until enough frames are in', () => {
    const estimator = new FrameRateEstimator();
    for (const t of timestamps(24, 6)) estimator.push(t);
    expect(estimator.value()).toBeNull();
    estimator.push(10 + 6 / 24);
    expect(estimator.value()).toBe(24);
  });

  it('measures 23.976, 25 and 60 fps', () => {
    for (const fps of [24000 / 1001, 25, 60]) {
      const estimator = new FrameRateEstimator();
      for (const t of timestamps(fps, 20)) estimator.push(t);
      expect(estimator.value()).toBe(snapFrameRate(fps));
    }
  });

  it('is not fooled by a dropped frame now and then', () => {
    const estimator = new FrameRateEstimator();
    const times = timestamps(30, 20).filter((_, i) => i % 7 !== 3);
    for (const t of times) estimator.push(t);
    expect(estimator.value()).toBe(30);
  });

  it('ignores seeks, loops and repeated timestamps', () => {
    const estimator = new FrameRateEstimator();
    for (const t of timestamps(25, 10, 100)) estimator.push(t);
    estimator.push(5); // seek back
    estimator.push(5); // same frame twice
    estimator.push(400); // seek forward
    for (const t of timestamps(25, 10, 400.04)) estimator.push(t);
    expect(estimator.value()).toBe(25);
    estimator.push(NaN);
    expect(estimator.value()).toBe(25);
  });

  it('forgets everything on reset', () => {
    const estimator = new FrameRateEstimator();
    for (const t of timestamps(24, 20)) estimator.push(t);
    estimator.reset();
    expect(estimator.value()).toBeNull();
    for (const t of timestamps(50, 20)) estimator.push(t);
    expect(estimator.value()).toBe(50);
  });
});
