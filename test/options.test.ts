import { describe, expect, it } from 'vitest';
import { DEFAULT_ICON, DEFAULT_LABELS, describeMode, resolveOptions } from '../src/options';
import { DEFAULT_TUNING } from '../src/plan';

describe('resolveOptions', () => {
  it('fills every default for no options', () => {
    const o = resolveOptions();
    expect(o.mode).toBe('auto');
    expect(o.labels).toEqual(DEFAULT_LABELS);
    expect(o.setting).toBe(true);
    expect(o.icon).toBe(DEFAULT_ICON);
    expect(o.compare).toBe(false);
    expect(o.frameBudgetMs).toBeNull();
    expect(o.slowFrameMs).toBeNull();
    expect(o.models).toEqual({});
    expect(o.tuning).toEqual(DEFAULT_TUNING);
    expect(o.autoDowngrade).toBe(true);
    expect(o.syncAudio).toBe(true);
    expect(o.maxOutputPixels).toBe(3840 * 2160);
    expect(o.cache).toBe(true);
    expect(o.cacheKey).toBe('artplayer-plugin-liveaction-sr');
    expect(o.onModeChange).toBeNull();
    expect(o.debug).toBe(false);
  });

  it('survives null, non-objects and garbage', () => {
    expect(resolveOptions(null).mode).toBe('auto');
    expect(resolveOptions('quality' as never).mode).toBe('auto');
    const o = resolveOptions({
      mode: 'insane' as never,
      frameBudgetMs: -3,
      slowFrameMs: NaN,
      maxOutputPixels: 'big' as never,
      onModeChange: 'nope' as never,
      icon: '   ',
    });
    expect(o.mode).toBe('auto');
    expect(o.frameBudgetMs).toBeNull();
    expect(o.slowFrameMs).toBeNull();
    expect(o.maxOutputPixels).toBe(3840 * 2160);
    expect(o.onModeChange).toBeNull();
    expect(o.icon).toBe(DEFAULT_ICON);
  });

  it('keeps valid values and clamps extreme ones', () => {
    const cb = () => {};
    const o = resolveOptions({
      mode: 'quality',
      frameBudgetMs: 5000,
      maxOutputPixels: 10,
      onModeChange: cb,
      setting: false,
      cache: false,
      autoDowngrade: false,
      syncAudio: false,
    });
    expect(o.mode).toBe('quality');
    expect(o.frameBudgetMs).toBe(100);
    expect(resolveOptions({ frameBudgetMs: 6 }).frameBudgetMs).toBe(6);
    expect(o.maxOutputPixels).toBe(320 * 180);
    expect(o.onModeChange).toBe(cb);
    expect(o.setting).toBe(false);
    expect(o.cache).toBe(false);
    expect(o.autoDowngrade).toBe(false);
    expect(o.syncAudio).toBe(false);
  });

  it('merges partial labels over the defaults and ignores empty ones', () => {
    const o = resolveOptions({
      labels: { setting: 'Hochskalierung', off: '', quality: 'Qualität' },
    });
    expect(o.labels.setting).toBe('Hochskalierung');
    expect(o.labels.quality).toBe('Qualität');
    expect(o.labels.off).toBe('Off');
    expect(o.labels.auto).toBe('Auto');
  });

  it('has a translatable notice for a preset that cannot keep up', () => {
    expect(resolveOptions().labels.struggling).toBe(
      'Enhancing is heavy for this GPU; try a lighter preset',
    );
    const o = resolveOptions({ labels: { struggling: 'Zu schwer für diese GPU' } });
    expect(o.labels.struggling).toBe('Zu schwer für diese GPU');
  });

  it('turns the split view on only for an explicit true', () => {
    expect(resolveOptions({ compare: true }).compare).toBe(true);
    expect(resolveOptions({ compare: 'yes' as never }).compare).toBe(false);
  });

  it('keeps the default icon unfilled inside ArtPlayer', () => {
    expect(DEFAULT_ICON).toContain('style="fill:none"');
  });

  it('does not share the defaults object between players', () => {
    const a = resolveOptions();
    a.labels.setting = 'changed';
    expect(resolveOptions().labels.setting).toBe('Enhance');
  });

  it('has a translatable notice for a video it may not read', () => {
    expect(resolveOptions().labels.blocked).toContain('CORS');
    expect(resolveOptions({ labels: { blocked: 'Gesperrt' } }).labels.blocked).toBe('Gesperrt');
  });

  it('keeps model sources for the CNN presets only', () => {
    const weights = { arch: 'x' };
    const loader = async () => weights;
    const o = resolveOptions({
      models: { balanced: '/weights/la-s.json', quality: loader, performance: 'x' } as never,
    });
    expect(o.models).toEqual({ balanced: '/weights/la-s.json', quality: loader });
    expect(resolveOptions({ models: { balanced: '', quality: null as never } }).models).toEqual({});
  });

  it('clamps tuning to 0..1 and keeps 0 so a stage can be turned off', () => {
    const o = resolveOptions({ tuning: { deband: 0, sharpness: 7, grain: NaN } });
    expect(o.tuning).toEqual({ deband: 0, sharpness: 1, grain: DEFAULT_TUNING.grain });
    expect(resolveOptions({ tuning: 'strong' as never }).tuning).toEqual(DEFAULT_TUNING);
  });
});

describe('describeMode', () => {
  it('names the explicit mode', () => {
    expect(describeMode('balanced', 'balanced', DEFAULT_LABELS, true)).toBe('Balanced');
  });

  it('shows what auto picked', () => {
    expect(describeMode('auto', 'quality', DEFAULT_LABELS, true)).toBe('Auto (Quality)');
    expect(describeMode('auto', 'off', DEFAULT_LABELS, true)).toBe('Auto (Off)');
  });

  it('uses translated labels', () => {
    const labels = {
      ...DEFAULT_LABELS,
      autoActive: 'Automatisch: {mode}',
      performance: 'Leistung',
    };
    expect(describeMode('auto', 'performance', labels, true)).toBe('Automatisch: Leistung');
  });

  it('says unsupported when WebGPU is missing', () => {
    expect(describeMode('auto', 'off', DEFAULT_LABELS, false)).toBe('Not supported');
  });
});
