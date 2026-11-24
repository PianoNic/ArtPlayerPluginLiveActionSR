import { describe, expect, it } from 'vitest';
import {
  effectivePreset,
  isMode,
  isModelPreset,
  median,
  menuModes,
  pickAutoMode,
  stepDown,
  type BenchmarkSample,
  type Preset,
} from '../src/modes';

const only = (...presets: Preset[]) => new Set<Preset>(presets);

describe('pickAutoMode', () => {
  const budget = 8;

  it('picks the strongest preset that fits the budget', () => {
    const samples: BenchmarkSample[] = [
      { preset: 'performance', ms: 1.2 },
      { preset: 'balanced', ms: 3.4 },
      { preset: 'quality', ms: 7.9 },
    ];
    expect(pickAutoMode(samples, budget)).toBe('quality');
  });

  it('stops below a preset that misses the budget', () => {
    const samples: BenchmarkSample[] = [
      { preset: 'performance', ms: 2 },
      { preset: 'balanced', ms: 6 },
      { preset: 'quality', ms: 14 },
    ];
    expect(pickAutoMode(samples, budget)).toBe('balanced');
  });

  it('treats the budget as inclusive', () => {
    expect(pickAutoMode([{ preset: 'performance', ms: 8 }], budget)).toBe('performance');
  });

  it('returns off when even the cheapest preset is too slow', () => {
    expect(pickAutoMode([{ preset: 'performance', ms: 9 }], budget)).toBe('off');
  });

  it('returns off for no samples, failed builds and NaN', () => {
    expect(pickAutoMode([], budget)).toBe('off');
    expect(pickAutoMode([{ preset: 'performance', ms: Infinity }], budget)).toBe('off');
    expect(pickAutoMode([{ preset: 'performance', ms: NaN }], budget)).toBe('off');
  });

  it('never picks a preset that was not measured, such as one without weights', () => {
    // Auto only benchmarks what is available: with no weights that is performance alone.
    expect(pickAutoMode([{ preset: 'performance', ms: 1 }], budget)).toBe('performance');
  });

  it('only considers presets that were measured (early stop after a miss)', () => {
    const samples: BenchmarkSample[] = [
      { preset: 'performance', ms: 3 },
      { preset: 'balanced', ms: 20 },
    ];
    expect(pickAutoMode(samples, budget)).toBe('performance');
  });

  it('does not depend on sample order', () => {
    const samples: BenchmarkSample[] = [
      { preset: 'quality', ms: 5 },
      { preset: 'performance', ms: 1 },
      { preset: 'balanced', ms: 2 },
    ];
    expect(pickAutoMode(samples, budget)).toBe('quality');
  });
});

describe('stepDown', () => {
  it('walks quality -> balanced -> performance -> off', () => {
    expect(stepDown('quality')).toBe('balanced');
    expect(stepDown('balanced')).toBe('performance');
    expect(stepDown('performance')).toBe('off');
    expect(stepDown('off')).toBe('off');
  });

  it('skips presets without weights', () => {
    const noBalanced = only('performance', 'quality');
    expect(stepDown('quality', noBalanced)).toBe('performance');
    expect(stepDown('quality', only('performance'))).toBe('performance');
  });

  it('treats performance as always available', () => {
    expect(stepDown('balanced', only())).toBe('performance');
  });
});

describe('menuModes', () => {
  it('offers auto, off and performance without any weights', () => {
    expect(menuModes(only('performance'))).toEqual(['auto', 'off', 'performance']);
    expect(menuModes(only())).toEqual(['auto', 'off', 'performance']);
  });

  it('adds a CNN preset once its weights are available, in strength order', () => {
    expect(menuModes(only('performance', 'quality'))).toEqual([
      'auto',
      'off',
      'performance',
      'quality',
    ]);
    expect(menuModes(only('performance', 'balanced', 'quality'))).toEqual([
      'auto',
      'off',
      'performance',
      'balanced',
      'quality',
    ]);
  });
});

describe('effectivePreset', () => {
  it('keeps an available preset', () => {
    expect(effectivePreset('balanced', only('performance', 'balanced'))).toBe('balanced');
  });

  it('falls back to the strongest available preset below', () => {
    expect(effectivePreset('quality', only('performance', 'balanced'))).toBe('balanced');
    expect(effectivePreset('quality', only('performance'))).toBe('performance');
    expect(effectivePreset('balanced', only('performance', 'quality'))).toBe('performance');
  });

  it('never falls back to nothing', () => {
    expect(effectivePreset('performance', only())).toBe('performance');
    expect(effectivePreset('quality', only())).toBe('performance');
  });
});

describe('isModelPreset', () => {
  it('is true for the presets that need weights', () => {
    expect(isModelPreset('balanced')).toBe(true);
    expect(isModelPreset('quality')).toBe(true);
    expect(isModelPreset('performance')).toBe(false);
    expect(isModelPreset('auto')).toBe(false);
  });
});

describe('median', () => {
  it('handles odd and even lengths without mutating the input', () => {
    const values = [5, 1, 3];
    expect(median(values)).toBe(3);
    expect(values).toEqual([5, 1, 3]);
    expect(median([4, 1, 3, 2])).toBe(2.5);
  });

  it('is Infinity for an empty list, so it can never pass a budget', () => {
    expect(median([])).toBe(Infinity);
  });
});

describe('isMode', () => {
  it('accepts the five modes and nothing else', () => {
    for (const mode of ['auto', 'off', 'performance', 'balanced', 'quality']) {
      expect(isMode(mode)).toBe(true);
    }
    expect(isMode('ultra')).toBe(false);
    expect(isMode(undefined)).toBe(false);
    expect(isMode(3)).toBe(false);
  });
});
