import { describe, expect, it } from 'vitest';
import {
  benchmarkCacheKey,
  CACHE_TTL_MS,
  readCachedMode,
  writeCachedMode,
  type StorageLike,
} from '../src/cache';

function memoryStorage(): StorageLike & { data: Map<string, string> } {
  const data = new Map<string, string>();
  return {
    data,
    getItem: (key) => data.get(key) ?? null,
    setItem: (key, value) => void data.set(key, value),
  };
}

const throwing: StorageLike = {
  getItem: () => {
    throw new Error('SecurityError');
  },
  setItem: () => {
    throw new Error('QuotaExceededError');
  },
};

const native = { width: 1280, height: 720 };
const screen = { width: 1920, height: 1080 };

describe('benchmark cache', () => {
  const presets = ['performance'];
  const key = benchmarkCacheKey('p', 'nvidia|ampere', native, screen, 12, presets);

  it('round-trips a decision', () => {
    const storage = memoryStorage();
    writeCachedMode(storage, key, 'balanced', 1000);
    expect(readCachedMode(storage, key, 2000)).toBe('balanced');
  });

  it('round-trips off', () => {
    const storage = memoryStorage();
    writeCachedMode(storage, key, 'off', 1000);
    expect(readCachedMode(storage, key, 2000)).toBe('off');
  });

  it('expires after the TTL and rejects entries from the future', () => {
    const storage = memoryStorage();
    writeCachedMode(storage, key, 'quality', 1000);
    expect(readCachedMode(storage, key, 1000 + CACHE_TTL_MS + 1)).toBeNull();
    expect(readCachedMode(storage, key, 500)).toBeNull();
  });

  it('treats malformed or foreign entries as a miss', () => {
    const storage = memoryStorage();
    storage.data.set(key, 'not json');
    expect(readCachedMode(storage, key)).toBeNull();
    storage.data.set(key, JSON.stringify({ v: 1, mode: 'insane', at: Date.now() }));
    expect(readCachedMode(storage, key)).toBeNull();
    storage.data.set(key, JSON.stringify({ v: 999, mode: 'quality', at: Date.now() }));
    expect(readCachedMode(storage, key)).toBeNull();
    storage.data.set(key, 'null');
    expect(readCachedMode(storage, key)).toBeNull();
  });

  it('never throws when storage is blocked or full', () => {
    expect(readCachedMode(throwing, key)).toBeNull();
    expect(() => writeCachedMode(throwing, key, 'quality')).not.toThrow();
    expect(readCachedMode(null, key)).toBeNull();
    expect(() => writeCachedMode(null, key, 'quality')).not.toThrow();
  });

  it('keys per GPU and per resolution, tolerating small window changes', () => {
    const at = (gpu: string, video: typeof native, target: typeof screen) =>
      benchmarkCacheKey('p', gpu, video, target, 12, presets);
    expect(at('intel|xe', native, screen)).not.toBe(key);
    expect(at('nvidia|ampere', native, { width: 1916, height: 1078 })).toBe(key);
    expect(at('nvidia|ampere', native, { width: 3840, height: 2160 })).not.toBe(key);
    expect(at('nvidia|ampere', screen, screen)).not.toBe(key);
  });

  it('keys per frame budget, so a 60 fps video does not reuse a 24 fps decision', () => {
    const tv = benchmarkCacheKey('p', 'nvidia|ampere', native, screen, 5.8333, presets);
    expect(tv).not.toBe(key);
    expect(benchmarkCacheKey('p', 'nvidia|ampere', native, screen, 12.0001, presets)).toBe(key);
  });

  it('keys per set of available presets, in any order', () => {
    const withModels = (list: string[]) =>
      benchmarkCacheKey('p', 'nvidia|ampere', native, screen, 12, list);
    expect(withModels(['performance', 'balanced'])).not.toBe(key);
    expect(withModels(['balanced', 'performance'])).toBe(withModels(['performance', 'balanced']));
  });
});
