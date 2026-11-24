import { describe, expect, it } from 'vitest';
import {
  debandRadius,
  debandThreshold,
  DEFAULT_TUNING,
  grainAmplitude,
  planPasses,
  rcasSharpness,
  type Tuning,
} from '../src/plan';

const p720 = { width: 1280, height: 720 };
const p1080 = { width: 1920, height: 1080 };
const p1440 = { width: 2560, height: 1440 };
const p2160 = { width: 3840, height: 2160 };

const kinds = (plan: ReturnType<typeof planPasses>) => plan.stages.map((stage) => stage.kind);

describe('planPasses: performance', () => {
  it('debands at the source size, upscales with EASU, then sharpens at the target', () => {
    const plan = planPasses('performance', p720, p1440, DEFAULT_TUNING, null);
    expect(kinds(plan)).toEqual(['deband', 'easu', 'rcas']);
    expect(plan.stages[0]).toEqual({ kind: 'deband', size: p720 });
    expect(plan.stages[1]).toEqual({ kind: 'easu', from: p720, size: p1440 });
    expect(plan.stages[2]).toEqual({ kind: 'rcas', size: p1440 });
    expect(plan.output).toEqual(p1440);
    expect(plan.upscaling).toBe(true);
    expect(plan.model).toBe(false);
  });

  it('runs the 1x cleanup when the screen is not larger than the video', () => {
    for (const target of [p1080, { width: 1900, height: 1070 }, p720]) {
      const plan = planPasses('performance', p1080, target, DEFAULT_TUNING, null);
      expect(kinds(plan)).toEqual(['deband', 'rcas']);
      expect(plan.output).toEqual(p1080);
      expect(plan.upscaling).toBe(false);
    }
  });

  it('skips EASU for a tiny size difference, but not for a real one on one axis', () => {
    const nearly = planPasses(
      'performance',
      p1080,
      { width: 1950, height: 1090 },
      DEFAULT_TUNING,
      null,
    );
    expect(kinds(nearly)).toEqual(['deband', 'rcas']);
    // A letterboxed source stretched wider: larger on one axis, the same on the other.
    const wider = planPasses(
      'performance',
      p1080,
      { width: 2400, height: 1080 },
      DEFAULT_TUNING,
      null,
    );
    expect(kinds(wider)).toEqual(['deband', 'easu', 'rcas']);
  });

  it('never runs EASU to downscale on either axis', () => {
    const plan = planPasses(
      'performance',
      p1080,
      { width: 2560, height: 1000 },
      DEFAULT_TUNING,
      null,
    );
    expect(kinds(plan)).toEqual(['deband', 'rcas']);
  });

  it('drops a stage whose strength is 0', () => {
    const none: Tuning = { deband: 0, sharpness: 0, grain: 0 };
    expect(kinds(planPasses('performance', p720, p1440, none, null))).toEqual(['easu']);
    expect(kinds(planPasses('performance', p1080, p1080, none, null))).toEqual([]);
    const noDeband = planPasses('performance', p720, p1440, { ...DEFAULT_TUNING, deband: 0 }, null);
    expect(kinds(noDeband)).toEqual(['easu', 'rcas']);
  });
});

describe('planPasses: CNN presets', () => {
  it('runs the CNN at 2x and lets EASU cover the rest', () => {
    const plan = planPasses('quality', p720, p2160, DEFAULT_TUNING, 2);
    expect(kinds(plan)).toEqual(['deband', 'model', 'easu', 'rcas']);
    expect(plan.stages[1]).toEqual({ kind: 'model', from: p720, size: p1440, scale: 2 });
    expect(plan.stages[2]).toEqual({ kind: 'easu', from: p1440, size: p2160 });
    expect(plan.model).toBe(true);
  });

  it('skips EASU when the CNN output already covers the target', () => {
    const exact = planPasses('balanced', p720, p1440, DEFAULT_TUNING, 2);
    expect(kinds(exact)).toEqual(['deband', 'model', 'rcas']);
    // 720p in a 1080p window: the 1440p CNN output is scaled down when presented.
    const over = planPasses('balanced', p720, p1080, DEFAULT_TUNING, 2);
    expect(kinds(over)).toEqual(['deband', 'model', 'rcas']);
    expect(over.output).toEqual(p1440);
  });

  it('does not run the CNN unless the screen is more than 1.2x the video on both axes', () => {
    const plan = planPasses('quality', p1080, { width: 2200, height: 1240 }, DEFAULT_TUNING, 2);
    expect(kinds(plan)).toEqual(['deband', 'easu', 'rcas']);
    expect(plan.model).toBe(false);
  });

  it('runs the performance stages without weights', () => {
    const withModel = planPasses('quality', p720, p1440, DEFAULT_TUNING, null);
    expect(withModel).toEqual(planPasses('performance', p720, p1440, DEFAULT_TUNING, null));
  });

  it('ignores weights for performance', () => {
    expect(kinds(planPasses('performance', p720, p1440, DEFAULT_TUNING, 2))).toEqual([
      'deband',
      'easu',
      'rcas',
    ]);
  });
});

describe('stage parameters', () => {
  it('maps deband strength to a threshold in signal units', () => {
    expect(debandThreshold(0)).toBe(0);
    expect(debandThreshold(0.5)).toBeCloseTo(3 / 255, 8);
    expect(debandThreshold(5)).toBeCloseTo(6 / 255, 8);
  });

  it('scales the deband radius with the video height', () => {
    expect(debandRadius(p1080)).toBe(16);
    expect(debandRadius(p720)).toBe(11);
    expect(debandRadius(p2160)).toBe(32);
    expect(debandRadius({ width: 320, height: 180 })).toBe(4);
  });

  it('maps sharpness to RCAS stops: 1 is full strength, 0.5 one stop down, 0 off', () => {
    expect(rcasSharpness(1)).toBe(1);
    expect(rcasSharpness(0.5)).toBe(0.5);
    expect(rcasSharpness(0)).toBe(0);
    expect(rcasSharpness(DEFAULT_TUNING.sharpness)).toBeLessThan(0.5);
    expect(rcasSharpness(DEFAULT_TUNING.sharpness)).toBeGreaterThan(0.25);
  });

  it('keeps the default grain light', () => {
    expect(grainAmplitude(0)).toBe(0);
    expect(grainAmplitude(DEFAULT_TUNING.grain)).toBeLessThanOrEqual(3 / 255);
    expect(grainAmplitude(10)).toBe(0.04);
  });
});
