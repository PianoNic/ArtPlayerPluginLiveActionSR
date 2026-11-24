import type Artplayer from 'artplayer';
import { AudioDelay, audioDelayFor, isIos, LagEstimator, MAX_AUDIO_DELAY_MS } from './audio.js';
import { benchmarkCacheKey, readCachedMode, safeLocalStorage, writeCachedMode } from './cache.js';
import { FrameRateEstimator, frameBudgetFor, slowFrameFor } from './framerate.js';
import { acquireGpu, measurePreset, Renderer, type GpuContext } from './gpu.js';
import { computeLayout, targetChanged, type Rect, type Size } from './layout.js';
import { isGraphModel, loadModel, type ModelWeights } from './model.js';
import {
  effectivePreset,
  isMode,
  isModelPreset,
  isPreset,
  median,
  menuModes,
  MODEL_PRESETS,
  pickAutoMode,
  PRESETS,
  stepDown,
  type ActiveMode,
  type BenchmarkSample,
  type Mode,
  type ModelPreset,
  type Preset,
} from './modes.js';
import { FrameMonitor, watchdogAction } from './monitor.js';
import { describeMode, resolveOptions, type LiveActionSrOptions } from './options.js';
import type { PassPlan } from './plan.js';

export type { LiveActionSrLabels, LiveActionSrOptions } from './options.js';
export type { ActiveMode, Mode, ModelPreset, Preset } from './modes.js';
export type {
  Conv3x3Layer,
  DepthToSpaceLayer,
  LinearModel,
  LumaCoefficients,
  ModelLayer,
  ModelSource,
  ModelWeights,
} from './model.js';
export type {
  Activation,
  GraphInput,
  GraphModel,
  GraphNode,
  GraphOutput,
  Padding,
} from './graph.js';
export type { Tuning } from './plan.js';
export { DEFAULT_LABELS } from './options.js';
export { MODES } from './modes.js';
export { isGraphModel, ModelError, parseModel } from './model.js';

export const PLUGIN_NAME = 'artplayerPluginLiveActionSR';
const SETTING_NAME = 'artplayer-plugin-liveaction-sr';
const LOG_PREFIX = '[liveaction-sr]';

export interface LiveActionSrPlugin {
  name: typeof PLUGIN_NAME;
  /** Select a mode. Unknown values are ignored; a CNN preset without weights selects `performance`. */
  setMode(mode: Mode): void;
  /** The selected mode, `auto` included. */
  getMode(): Mode;
  /** What is rendering right now: auto's pick, auto's downgraded preset, or `off`. */
  getActiveMode(): ActiveMode;
  /** The presets that can run here: `performance`, plus every CNN preset whose weights are usable. */
  getAvailablePresets(): Preset[];
  /**
   * Split view: the original video left of `position` (0..1 of the width), the enhanced one right
   * of it. `setCompare(false)` shows the enhanced frame in full again.
   */
  setCompare(on: boolean, position?: number): void;
  getCompare(): boolean;
  /** What is rendering and how fast, for a status readout. */
  getStats(): LiveActionSrStats;
  /** WebGPU is usable in this browser. `false` until the capability check has finished. */
  readonly supported: boolean;
  /** Resolves with `supported` once the capability check has finished. Never rejects. */
  readonly ready: Promise<boolean>;
  /** Stop rendering and remove the canvas and the settings entry. Called on `art.destroy()` too. */
  destroy(): void;
}

export interface LiveActionSrStats {
  active: ActiveMode;
  /** Average GPU time per frame over the last frames, in ms; null before the first frame. */
  frameMs: number | null;
  /** The video's own resolution. */
  native: Size | null;
  /** The resolution the pipeline renders to (on-screen size times the device pixel ratio). */
  target: Size | null;
  /** Whether a stage raises the resolution (EASU or the CNN); false for the 1x cleanup. */
  upscaling: boolean;
  /** Whether the luma CNN runs. */
  model: boolean;
  /** The stages running, in order, e.g. `['deband', 'easu', 'rcas']`. */
  stages: string[];
  /** The video's frame rate as measured; null until known. */
  fps: number | null;
  /** Auto's frame budget for this video, in ms. */
  budgetMs: number;
  /** The picked preset keeps falling behind on this GPU; it still renders, the viewer was told. */
  struggling: boolean;
  /** How much later than the plain video the enhanced frame shows, smoothed; null when unknown. */
  canvasLagMs: number | null;
  /** How far the audio is delayed to match it (`syncAudio`), mid-ramp included. */
  audioDelayMs: number;
}

interface VideoFrameMetadata {
  mediaTime?: number;
}

type VideoFrameCallbackVideo = HTMLVideoElement & {
  requestVideoFrameCallback?: (
    callback: (now: number, metadata: VideoFrameMetadata) => void,
  ) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
};

function isSecurityError(error: unknown): boolean {
  return (
    typeof error === 'object' &&
    error !== null &&
    (error as { name?: unknown }).name === 'SecurityError'
  );
}

export default function artplayerPluginLiveActionSR(option: LiveActionSrOptions = {}) {
  return (art: Artplayer): LiveActionSrPlugin => {
    const opts = resolveOptions(option);
    const video = art.template.$video as VideoFrameCallbackVideo;
    const storage = opts.cache ? safeLocalStorage() : null;
    const log = (...args: unknown[]) => {
      if (opts.debug) console.info(LOG_PREFIX, ...args);
    };

    // `performance` always; a CNN preset while its weights are configured and have not failed.
    const available = new Set<Preset>([
      'performance',
      ...MODEL_PRESETS.filter((preset) => opts.models[preset] !== undefined),
    ]);
    const weights = new Map<ModelPreset, Promise<ModelWeights | null>>();
    const fps = new FrameRateEstimator();

    /** A preset that cannot run falls back to the strongest available one below it. */
    const normalize = (mode: Mode): Mode =>
      isPreset(mode) ? effectivePreset(mode, available) : mode;

    let selected: Mode = normalize(opts.mode);
    let active: ActiveMode = 'off';
    let supported = false;
    let destroyed = false;
    let gpu: GpuContext | null = null;
    let renderer: Renderer | null = null;
    // Set by a downgrade; cleared when the source or the selection changes.
    let ceiling: ActiveMode | null = null;
    // A picked preset cannot keep up; the notice shows once, until the source or selection changes.
    let struggling = false;
    // A cross-origin source WebGPU may not read; not retried.
    let blockedSource: string | null = null;
    // Bumped by every rebuild; async work from an older generation stops.
    let generation = 0;
    let built: { native: Size; target: Size; plan: PassPlan } | null = null;
    let settingAdded = false;
    let compare = opts.compare;
    let comparePosition = 0.5;
    let frameMs: number | null = null;

    const canvas = document.createElement('canvas');
    canvas.className = 'art-liveaction-sr';
    Object.assign(canvas.style, {
      position: 'absolute',
      zIndex: '10',
      pointerEvents: 'none',
      display: 'none',
    });
    video.insertAdjacentElement('afterend', canvas);

    // The split line of the compare view.
    const divider = document.createElement('div');
    divider.className = 'art-liveaction-sr-divider';
    Object.assign(divider.style, {
      position: 'absolute',
      zIndex: '10',
      width: '2px',
      marginLeft: '-1px',
      background: 'rgba(255, 255, 255, 0.85)',
      pointerEvents: 'none',
      display: 'none',
    });
    canvas.insertAdjacentElement('afterend', divider);

    function applyCompare(): void {
      const pct = `${(comparePosition * 100).toFixed(2)}%`;
      canvas.style.clipPath = compare ? `inset(0 0 0 ${pct})` : '';
      const shown = compare && canvas.style.display !== 'none';
      divider.style.display = shown ? 'block' : 'none';
      if (shown) {
        const left = parseFloat(canvas.style.left) || 0;
        const width = parseFloat(canvas.style.width) || 0;
        Object.assign(divider.style, {
          left: `${left + width * comparePosition}px`,
          top: canvas.style.top,
          height: canvas.style.height,
        });
      }
    }
    applyCompare();

    const monitor = new FrameMonitor({ slowFrameMs: opts.slowFrameMs ?? slowFrameFor(null) });

    // ---- frame rate ---------------------------------------------------------------------------
    //
    // A 24 fps film leaves 41.7 ms per frame, 60 fps TV 16.7 ms. Auto's budget and the watchdog's
    // idea of a slow frame follow the measured rate unless the options fix them.

    function budget(): number {
      return opts.frameBudgetMs ?? frameBudgetFor(fps.value());
    }

    /** Feeds a presented frame's media time to the frame-rate estimate and the watchdog. */
    function trackFrameRate(mediaTime: number | undefined): void {
      if (typeof mediaTime !== 'number') return;
      fps.push(mediaTime);
      if (opts.slowFrameMs === null) {
        const slow = slowFrameFor(fps.value());
        if (slow !== monitor.slowFrameMs()) monitor.setSlowFrameMs(slow);
      }
    }

    /**
     * Watches a few frames of a playing video to learn its frame rate before auto benchmarks.
     * Returns at once when the video is paused, the rate is known, or the budget is fixed.
     */
    function probeFrameRate(gen: number, frames = 8, timeoutMs = 700): Promise<void> {
      return new Promise((resolve) => {
        const request = video.requestVideoFrameCallback?.bind(video);
        if (opts.frameBudgetMs !== null || fps.value() !== null || video.paused || !request) {
          resolve();
          return;
        }
        let seen = 0;
        let handle = 0;
        const timer = setTimeout(() => {
          video.cancelVideoFrameCallback?.(handle);
          resolve();
        }, timeoutMs);
        const step = (_now: number, metadata: VideoFrameMetadata) => {
          trackFrameRate(metadata?.mediaTime);
          if (++seen >= frames || fps.value() !== null || gen !== generation || destroyed) {
            clearTimeout(timer);
            resolve();
            return;
          }
          handle = request(step);
        };
        handle = request(step);
      });
    }

    // ---- model weights ------------------------------------------------------------------------

    /** The weights of a CNN preset, loaded once. Null, and the preset withdrawn, when they fail. */
    function loadWeights(preset: ModelPreset): Promise<ModelWeights | null> {
      let promise = weights.get(preset);
      if (!promise) {
        const source = opts.models[preset];
        promise = (
          source === undefined ? Promise.reject(new Error('no weights')) : loadModel(source)
        )
          .then((model) => {
            const size = isGraphModel(model)
              ? `${model.nodes.length} nodes`
              : `${model.layers.length} layers`;
            log('weights', preset, `${model.arch}, x${model.scale}, ${size}`);
            return model;
          })
          .catch((error: unknown) => {
            console.warn(`${LOG_PREFIX} ${preset} is unavailable; its weights failed:`, error);
            available.delete(preset);
            if (selected === preset) selected = normalize(selected);
            updateSetting();
            return null;
          });
        weights.set(preset, promise);
      }
      return promise;
    }

    /** What a preset runs: itself with its weights, or the strongest preset below it that can. */
    async function resolveRun(
      preset: Preset,
    ): Promise<{ preset: Preset; model: ModelWeights | null }> {
      let next = effectivePreset(preset, available);
      while (isModelPreset(next)) {
        const model = await loadWeights(next);
        if (model) return { preset: next, model };
        next = effectivePreset(next, available);
      }
      return { preset: 'performance', model: null };
    }

    // ---- audio sync ---------------------------------------------------------------------------
    //
    // The enhanced frame reaches the screen after the plain one would have: the video frame first,
    // then the GPU work. The audio is delayed by that much through Web Audio so the two stay in step.
    // It only starts once a frame was processed, which proves the video is readable (same-origin or
    // CORS): Web Audio would play a tainted element as silence.

    const lag = new LagEstimator();
    const audio =
      opts.syncAudio && typeof AudioContext === 'function' && !isIos(navigator)
        ? new AudioDelay(video)
        : null;

    function syncAudio(): void {
      if (!audio) return;
      if (active === 'off') return audio.set(0);
      const lagMs = lag.value();
      // Right after a rebuild the estimator is empty: the delay stays until it has refilled.
      if (lagMs === null) return;
      // Delaying the sound by a quarter second or more is no fix: that preset is too heavy.
      if (lagMs > MAX_AUDIO_DELAY_MS && selected !== 'auto') warnStruggling();
      audio.set(audioDelayFor(lagMs));
    }

    const wakeAudio = () => audio?.wake();

    // ---- layout -------------------------------------------------------------------------------

    function measure(): { rect: Rect; target: Size } | null {
      if (!video.videoWidth || !video.videoHeight) return null;
      const cs = getComputedStyle(video);
      const px = (value: string) => parseFloat(value) || 0;
      const box: Rect = {
        left: video.offsetLeft + px(cs.borderLeftWidth) + px(cs.paddingLeft),
        top: video.offsetTop + px(cs.borderTopWidth) + px(cs.paddingTop),
        width: video.clientWidth - px(cs.paddingLeft) - px(cs.paddingRight),
        height: video.clientHeight - px(cs.paddingTop) - px(cs.paddingBottom),
      };
      if (box.width <= 0 || box.height <= 0) return null;
      const layout = computeLayout(
        box,
        { width: video.videoWidth, height: video.videoHeight },
        cs.objectFit || 'contain',
        window.devicePixelRatio || 1,
        opts.maxOutputPixels,
      );
      Object.assign(canvas.style, {
        left: `${layout.rect.left}px`,
        top: `${layout.rect.top}px`,
        width: `${layout.rect.width}px`,
        height: `${layout.rect.height}px`,
        transform: cs.transform === 'none' ? '' : cs.transform,
      });
      applyCompare();
      return layout;
    }

    // ---- state --------------------------------------------------------------------------------

    function setActive(next: ActiveMode): void {
      const changed = next !== active;
      active = next;
      if (next === 'off') {
        hideCanvas();
        lag.reset();
        syncAudio();
      }
      if (changed) {
        log('active mode', next);
        notify();
      }
      updateSetting();
    }

    function notify(): void {
      try {
        opts.onModeChange?.(selected, active);
      } catch (error) {
        console.error(error);
      }
    }

    function fail(error: unknown): void {
      generation++;
      stopLoop();
      renderer?.release();
      built = null;
      setActive('off');
      if (isSecurityError(error)) {
        blockedSource = video.currentSrc;
        console.warn(
          `${LOG_PREFIX} The video is cross-origin without CORS, so WebGPU cannot read its ` +
            'frames. Serve it from the same origin, or with Access-Control-Allow-Origin and ' +
            'crossorigin="anonymous" on the video.',
        );
        try {
          art.notice.show = opts.labels.blocked;
        } catch {
          // Notice layer not available on this player.
        }
      } else {
        console.warn(`${LOG_PREFIX} Enhancing stopped:`, error);
      }
      try {
        opts.onError?.(error);
      } catch (callbackError) {
        console.error(callbackError);
      }
    }

    function hideCanvas(): void {
      canvas.style.display = 'none';
      frameMs = null;
      applyCompare();
    }

    // ---- decision -----------------------------------------------------------------------------

    async function benchmark(
      native: Size,
      target: Size,
      budgetMs: number,
      gen: number,
    ): Promise<ActiveMode | null> {
      if (!gpu) return 'off';
      const samples: BenchmarkSample[] = [];
      for (const preset of PRESETS) {
        if (!available.has(preset)) continue;
        let ms = Infinity;
        try {
          const model = isModelPreset(preset) ? await loadWeights(preset) : null;
          if (gen !== generation || destroyed) return null;
          // Weights that failed to load: the preset is gone, not slow.
          if (isModelPreset(preset) && !model) continue;
          const chain = { tuning: opts.tuning, model };
          ms = median(await measurePreset(gpu, video, preset, native, target, chain));
        } catch (error) {
          if (isSecurityError(error)) throw error;
          log('benchmark failed for', preset, error);
        }
        if (gen !== generation || destroyed) return null;
        samples.push({ preset, ms });
        log('benchmark', preset, `${ms.toFixed(2)} ms of ${budgetMs.toFixed(1)} ms`);
        // Stronger presets are slower still.
        if (!(ms <= budgetMs)) break;
      }
      return pickAutoMode(samples, budgetMs);
    }

    function cacheKey(native: Size, target: Size): string {
      return benchmarkCacheKey(
        opts.cacheKey,
        gpu?.name ?? 'unknown',
        native,
        target,
        budget(),
        PRESETS.filter((preset) => available.has(preset)),
      );
    }

    /** Works out what should render for the current selection and source, then builds it. */
    async function apply(): Promise<void> {
      const gen = ++generation;
      if (destroyed) return;
      if (!supported || selected === 'off' || !gpu) {
        stopLoop();
        renderer?.release();
        built = null;
        setActive('off');
        return;
      }
      if (blockedSource !== null && blockedSource === video.currentSrc) {
        setActive('off');
        return;
      }
      const layout = measure();
      // Not laid out yet; `loadedmetadata` or the resize observer calls back.
      if (!layout) return;
      const native = { width: video.videoWidth, height: video.videoHeight };
      const target = layout.target;

      let next: ActiveMode;
      let plan: PassPlan;
      try {
        if (selected === 'auto') {
          // Needs a decoded frame; `loadeddata` calls back.
          if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
          await probeFrameRate(gen);
          if (gen !== generation || destroyed) return;
          const key = cacheKey(native, target);
          const cached = readCachedMode(storage, key);
          if (cached) {
            next = cached;
            log('auto (cached)', cached);
          } else {
            const picked = await benchmark(native, target, budget(), gen);
            if (picked === null) return;
            next = picked;
            writeCachedMode(storage, key, next);
          }
        } else {
          next = selected;
        }
        if (ceiling && PRESETS.indexOf(next as never) > PRESETS.indexOf(ceiling as never)) {
          next = ceiling;
        }
        if (next === 'off') {
          stopLoop();
          renderer?.release();
          built = null;
          setActive('off');
          return;
        }
        const run = await resolveRun(next);
        if (gen !== generation || destroyed) return;
        next = run.preset;
        renderer ??= new Renderer(gpu, canvas);
        canvas.width = target.width;
        canvas.height = target.height;
        const chain = { tuning: opts.tuning, model: run.model };
        plan = await renderer.build(next, native, target, chain);
        if (gen !== generation || destroyed) return;
        log('stages', plan.stages.map((stage) => stage.kind).join(' > ') || 'none');
      } catch (error) {
        if (gen === generation) fail(error);
        return;
      }
      built = { native, target, plan };
      restartMonitor();
      lag.reset();
      frameMs = null;
      firstFrame = true;
      setActive(next);
      startLoop();
    }

    /** Tells the viewer once that their preset is too heavy, and keeps rendering it. */
    function warnStruggling(): void {
      if (struggling) return;
      struggling = true;
      log('struggling on', active, `bad ratio ${monitor.badRatio().toFixed(2)}`);
      try {
        art.notice.show = opts.labels.struggling;
      } catch {
        // Notice layer not available on this player.
      }
    }

    function downgrade(): void {
      const next = stepDown(active, available);
      log('downgrade', active, '->', next, `bad ratio ${monitor.badRatio().toFixed(2)}`);
      ceiling = next;
      if (selected === 'auto' && built)
        writeCachedMode(storage, cacheKey(built.native, built.target), next);
      void apply();
    }

    // ---- frame loop ---------------------------------------------------------------------------

    let loopHandle: number | null = null;
    let loopUsesVideoCallback = false;
    let busy = false;
    let firstFrame = true;
    let lastTime = -1;
    let lastDropped = 0;
    let framesSinceDropCheck = 0;
    // The video size the frame loop last asked a rebuild for.
    let rebuildRequested: string | null = null;

    function schedule(): void {
      if (destroyed || loopHandle !== null) return;
      if (typeof video.requestVideoFrameCallback === 'function') {
        loopUsesVideoCallback = true;
        loopHandle = video.requestVideoFrameCallback(onFrame);
      } else {
        loopUsesVideoCallback = false;
        loopHandle = requestAnimationFrame((now) => onFrame(now));
      }
    }

    function stopLoop(): void {
      if (loopHandle === null) return;
      if (loopUsesVideoCallback) video.cancelVideoFrameCallback?.(loopHandle);
      else cancelAnimationFrame(loopHandle);
      loopHandle = null;
    }

    function startLoop(): void {
      stopLoop();
      lastTime = -1;
      lastDropped = droppedFrames();
      framesSinceDropCheck = 0;
      // A paused video presents no new frames: draw the current one now. `onFrame` schedules the
      // next one itself; scheduling here as well would start a second loop.
      if (video.paused) onFrame(performance.now());
      else schedule();
    }

    function droppedFrames(): number {
      try {
        return video.getVideoPlaybackQuality?.().droppedVideoFrames ?? 0;
      } catch {
        return 0;
      }
    }

    function onFrame(frameStart = performance.now(), metadata?: VideoFrameMetadata): void {
      loopHandle = null;
      if (destroyed || active === 'off' || !renderer?.ready) return;
      schedule();
      if (!video.paused) trackFrameRate(metadata?.mediaTime);

      if (document.visibilityState === 'hidden') return;
      if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
      // A quality switch changes the frame size before `resize` fires. The old chain cannot draw
      // the new frames, so the plain video shows until the rebuild is in, not a frozen picture.
      // Asked once per size: re-arming the debounce on every frame kept it from ever firing.
      if (nativeChanged()) {
        const size = `${video.videoWidth}x${video.videoHeight}`;
        if (rebuildRequested !== size) {
          rebuildRequested = size;
          hideCanvas();
          onLayoutChange();
        }
        return;
      }
      if (video.paused && video.currentTime === lastTime && !firstFrame) return;
      if (busy) {
        if (!video.paused) monitor.recordSkipped();
        return;
      }
      lastTime = video.currentTime;

      if (!monitor.settled()) {
        // Drops from the start, a seek or a stall must not show up in the first count after it.
        lastDropped = droppedFrames();
        framesSinceDropCheck = 0;
      } else if (++framesSinceDropCheck >= 30) {
        framesSinceDropCheck = 0;
        const dropped = droppedFrames();
        if (dropped > lastDropped && !video.paused && !document.hidden)
          monitor.recordDropped(dropped - lastDropped);
        lastDropped = dropped;
      }

      const gen = generation;
      busy = true;
      renderer
        .render(video)
        .then((ms) => {
          busy = false;
          if (gen !== generation || destroyed) return;
          if (firstFrame) {
            firstFrame = false;
            canvas.style.display = 'block';
            applyCompare();
          }
          frameMs = frameMs === null ? ms : frameMs * 0.9 + ms * 0.1;
          if (!video.paused) {
            monitor.recordFrame(ms);
            // `frameStart` is when the compositor began the frame that brought this video frame; the
            // enhanced one can only start down the same path once the GPU is done with it.
            lag.push(performance.now() - frameStart);
            syncAudio();
          }
          const action = watchdogAction(
            selected,
            active,
            monitor.badWindows(),
            opts.autoDowngrade,
            available,
          );
          if (action === 'step') downgrade();
          else if (action === 'warn') warnStruggling();
        })
        .catch((error: unknown) => {
          busy = false;
          if (gen === generation && !destroyed) fail(error);
        });
    }

    // ---- reacting to the player ---------------------------------------------------------------

    let resizeTimer: ReturnType<typeof setTimeout> | null = null;
    function onLayoutChange(): void {
      if (resizeTimer) clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        resizeTimer = null;
        if (destroyed || !supported) return;
        const layout = measure();
        if (!layout) return;
        if (!built) {
          if (selected !== 'off') void apply();
        } else if (nativeChanged() || targetChanged(built.target, layout.target)) {
          void apply();
        }
      }, 150);
    }

    function onSourceStart(): void {
      // New source: hide the old frame and forget its downgrade and frame rate.
      hideCanvas();
      ceiling = null;
      struggling = false;
      firstFrame = true;
      fps.reset();
      if (opts.slowFrameMs === null) monitor.setSlowFrameMs(slowFrameFor(null));
    }

    /**
     * Starts the watchdog over, with a fresh warm-up and grace period. Called after every build and
     * on the playback events that drop frames on any GPU (start, seek, stall, tab switch), so those
     * never count against the preset.
     */
    function restartMonitor(): void {
      monitor.reset();
      lastDropped = droppedFrames();
      framesSinceDropCheck = 0;
    }

    function nativeChanged(): boolean {
      return (
        !built ||
        built.native.width !== video.videoWidth ||
        built.native.height !== video.videoHeight
      );
    }

    function onSourceReady(): void {
      if (!supported) return;
      if (nativeChanged()) {
        void apply();
      } else if (active !== 'off') {
        startLoop();
      } else if (selected !== 'off') {
        void apply();
      }
    }

    function onVisibility(): void {
      if (document.visibilityState === 'visible') {
        restartMonitor();
        if (active !== 'off') startLoop();
      }
    }

    const videoEvents: [string, () => void][] = [
      ['loadstart', onSourceStart],
      ['loadedmetadata', onSourceReady],
      ['loadeddata', onSourceReady],
      ['resize', onLayoutChange],
      ['play', onVisibility],
      ['play', wakeAudio],
      ['playing', restartMonitor],
      ['seeking', restartMonitor],
      ['seeked', restartMonitor],
      ['waiting', restartMonitor],
    ];
    for (const [event, handler] of videoEvents) video.addEventListener(event, handler);
    document.addEventListener('visibilitychange', onVisibility);
    // An audio context only starts after a user gesture.
    for (const event of ['pointerdown', 'keydown']) {
      document.addEventListener(event, wakeAudio, { capture: true, passive: true });
    }

    const resizeObserver =
      typeof ResizeObserver === 'function' ? new ResizeObserver(onLayoutChange) : null;
    resizeObserver?.observe(video);
    art.on('resize', onLayoutChange);
    art.on('fullscreen', onLayoutChange);
    art.on('fullscreenWeb', onLayoutChange);
    art.on('aspectRatio', onLayoutChange);
    art.on('flip', onLayoutChange);

    // ---- settings menu ------------------------------------------------------------------------

    function buildSetting() {
      return {
        name: SETTING_NAME,
        html: opts.labels.setting,
        icon: opts.icon,
        tooltip: describeMode(selected, active, opts.labels, supported),
        selector: menuModes(available).map((mode) => ({
          html: opts.labels[mode],
          value: mode,
          default: mode === selected,
        })),
        onSelect(item: { value?: unknown }) {
          if (isMode(item.value)) setMode(item.value);
          return describeMode(selected, active, opts.labels, supported);
        },
      };
    }

    function updateSetting(): void {
      if (!settingAdded || destroyed) return;
      try {
        art.setting.update(buildSetting());
      } catch {
        // Settings panel not available on this player.
      }
    }

    function addSetting(): void {
      if (!opts.setting || !supported || settingAdded) return;
      try {
        art.setting.add(buildSetting());
        settingAdded = true;
      } catch {
        // `setting: false` on the player, or an ArtPlayer without the settings component.
      }
    }

    // ---- public API ---------------------------------------------------------------------------

    function setMode(mode: Mode): void {
      if (destroyed || !isMode(mode)) return;
      const next = normalize(mode);
      if (next !== mode) log(mode, 'has no weights; selecting', next);
      const changed = next !== selected;
      selected = next;
      ceiling = null;
      struggling = false;
      if (changed) notify();
      updateSetting();
      void apply();
    }

    function destroy(): void {
      if (destroyed) return;
      destroyed = true;
      generation++;
      stopLoop();
      if (resizeTimer) clearTimeout(resizeTimer);
      for (const [event, handler] of videoEvents) video.removeEventListener(event, handler);
      document.removeEventListener('visibilitychange', onVisibility);
      for (const event of ['pointerdown', 'keydown']) {
        document.removeEventListener(event, wakeAudio, { capture: true });
      }
      // The element stays routed through Web Audio for good; only the delay goes.
      audio?.reset();
      resizeObserver?.disconnect();
      if (settingAdded) {
        try {
          art.setting.remove(SETTING_NAME);
        } catch {
          // Player already torn down.
        }
      }
      renderer?.destroy();
      renderer = null;
      gpu?.device.destroy();
      gpu = null;
      canvas.remove();
      divider.remove();
    }

    art.on('destroy', destroy);

    const ready = acquireGpu().then((context) => {
      if (destroyed) {
        context?.device.destroy();
        return false;
      }
      gpu = context;
      supported = context !== null;
      log(
        supported
          ? `WebGPU on ${context?.name}${context?.f16 ? ' (shader-f16)' : ''}`
          : 'WebGPU unavailable; staying off',
      );
      if (context) {
        void context.device.lost.then((info) => {
          if (destroyed || info.reason === 'destroyed') return;
          gpu = null;
          renderer = null;
          supported = false;
          fail(new Error(`GPU device lost: ${info.message}`));
        });
      }
      addSetting();
      if (supported) void apply();
      else setActive('off');
      return supported;
    });

    return {
      name: PLUGIN_NAME,
      setMode,
      getMode: () => selected,
      getActiveMode: () => active,
      getAvailablePresets: () => PRESETS.filter((preset) => available.has(preset)),
      setCompare(on: boolean, position?: number) {
        compare = on === true;
        if (typeof position === 'number' && Number.isFinite(position))
          comparePosition = Math.min(1, Math.max(0, position));
        applyCompare();
      },
      getCompare: () => compare,
      getStats(): LiveActionSrStats {
        const native = built?.native ?? null;
        const target = built?.target ?? null;
        const plan = active === 'off' ? null : (built?.plan ?? null);
        return {
          active,
          frameMs: active === 'off' ? null : frameMs,
          native,
          target,
          upscaling: plan?.upscaling ?? false,
          model: plan?.model ?? false,
          stages: plan?.stages.map((stage) => stage.kind) ?? [],
          fps: fps.value(),
          budgetMs: budget(),
          struggling: struggling && active !== 'off',
          canvasLagMs: active === 'off' ? null : lag.value(),
          audioDelayMs: audio?.currentMs ?? 0,
        };
      },
      get supported() {
        return supported;
      },
      ready,
      destroy,
    };
  };
}

export { artplayerPluginLiveActionSR };
