<p align="center">
  <img src="assets/icon.svg" width="180" alt="ArtPlayerPluginLiveActionSR Logo" />
</p>

<h1 align="center">ArtPlayerPluginLiveActionSR</h1>

<p align="center">
  <strong>Real-time upscaling of live-action video for ArtPlayer, on WebGPU.</strong>
</p>

<p align="center">
  <a href="https://github.com/PianoNic/ArtPlayerPluginLiveActionSR"><img src="https://badgetrack.pianonic.ch/badge?tag=artplayerpluginliveactionsr&label=visits&color=0d1117&style=flat" alt="visits" /></a>
  <a href="https://www.npmjs.com/package/artplayer-plugin-liveaction-sr"><img src="https://img.shields.io/npm/v/artplayer-plugin-liveaction-sr?color=0d1117&label=npm" alt="npm version" /></a>
  <a href="https://www.npmjs.com/package/artplayer-plugin-liveaction-sr"><img src="https://img.shields.io/npm/dm/artplayer-plugin-liveaction-sr?color=0d1117&label=downloads" alt="npm downloads" /></a>
  <a href="https://github.com/PianoNic/ArtPlayerPluginLiveActionSR/blob/main/LICENSE"><img src="https://img.shields.io/github/license/PianoNic/ArtPlayerPluginLiveActionSR?color=0d1117" alt="MIT" /></a>
  <img src="https://img.shields.io/badge/WebGPU-0d1117.svg" alt="WebGPU" />
  <img src="https://img.shields.io/badge/TypeScript-0d1117.svg" alt="TypeScript" />
</p>

---

> **Heads up:** early development, pre-1.0 - options may still change between minor versions.
> Enhancing needs **WebGPU** (Chrome / Edge 113+, Safari 26+, Firefox 141+ on Windows) and a video
> that is **same-origin or served with CORS**. Everywhere else the plugin stays off quietly and the
> player works exactly as before.

## What is it?

An [ArtPlayer](https://artplayer.org) plugin that cleans up and upscales **films and TV** on every
frame, in the browser, on the GPU. It is the live-action sibling of
[ArtPlayerPluginAnime4K](https://github.com/PianoNic/ArtPlayerPluginAnime4K): same player
integration, a different picture pipeline.

Anime upscalers win by reconstructing clean line art. Live-action streams suffer from codec damage
instead - banding in skies, smeared texture, blocking - and sharpening that damage only makes it
more visible. So the pipeline cleans before it upscales, sharpens gently, and puts back the film
grain that compression ate:

1. **Deband** at the source resolution - an f3kdb-style filter written for this project.
2. **AMD FSR 1 EASU** - edge-adaptive upscale to the on-screen size.
3. **AMD FSR 1 RCAS** - sharpening at a low strength, with its noise detection on.
4. **Film grain** - light, luma-only, moving grain at the output resolution, added last.

All of it is hand-written WGSL with no runtime dependencies. The `balanced` and `quality` presets
add a small **luma CNN** that predicts a residual on top of a bilinear 2x upscale; its weights are
trained separately and plugged in through `models`. The reasoning behind every choice - which
models and shaders were ruled out and why, licences, budgets - is in
[docs/research.md](docs/research.md).

## Features

- **Five modes** - `auto`, `off`, `performance`, `balanced`, `quality`, switchable from the settings gear or the API
- **No weights needed to start** - `performance` is pure shader work; the CNN presets only appear once their weights are configured
- **Auto mode** - benchmarks each available preset on the actual video and screen size, picks the strongest one that fits the frame budget, and remembers the result per GPU
- **Frame-rate aware** - measures the video's frame rate, so a 24 fps film gets more GPU time per frame than 60 fps TV
- **Runtime downgrade** - in auto, watches GPU time, skipped and dropped frames and steps down one preset when the GPU cannot keep up; a preset the viewer picked is never lowered, they get a one-time notice instead
- **1x cleanup** - when the screen is not larger than the video, it still debands, sharpens and adds grain
- **Audio stays in sync** - the enhanced picture reaches the screen a little after the plain video would have; the audio is delayed by exactly that much
- **Graceful fallback** - no WebGPU, no adapter, a cross-origin video, missing weights or a lost GPU all end in `off` or `performance`, never in an exception
- **Layered, not replaced** - a `<canvas>` over the `<video>`; playback, audio, seeking, subtitles and fullscreen are untouched
- **Coexists with Anime4K** - its own settings entry, canvas class and storage keys, so both plugins can sit on one page
- **Translatable** - every label comes in through `labels`

## Installation

```bash
npm install artplayer-plugin-liveaction-sr
```

or `bun add artplayer-plugin-liveaction-sr`. ArtPlayer 5 is a peer dependency.

## Usage

```ts
import Artplayer from 'artplayer';
import artplayerPluginLiveActionSR from 'artplayer-plugin-liveaction-sr';

const art = new Artplayer({
  container: '#player',
  url: '/videos/film.m3u8',
  setting: true, // the mode picker lives in the settings menu
  moreVideoAttr: { crossOrigin: 'anonymous' }, // for a video on another origin, see CORS below
  plugins: [artplayerPluginLiveActionSR({ mode: 'auto' })],
});

const sr = art.plugins.artplayerPluginLiveActionSR;
sr.setMode('performance');
```

Without a bundler, load the browser build after ArtPlayer; it defines `artplayerPluginLiveActionSR`:

```html
<script src="https://cdn.jsdelivr.net/npm/artplayer/dist/artplayer.js"></script>
<script src="https://cdn.jsdelivr.net/npm/artplayer-plugin-liveaction-sr"></script>
<script>
  const art = new Artplayer({
    container: '#player',
    url: 'film.mp4',
    setting: true,
    plugins: [artplayerPluginLiveActionSR()],
  });
</script>
```

## Presets

| Mode          | Pipeline                                                         | Needs             |
| ------------- | ---------------------------------------------------------------- | ----------------- |
| `off`         | -                                                                | -                 |
| `performance` | deband > FSR1 EASU to the screen > RCAS > grain                  | WebGPU            |
| `balanced`    | deband > small luma CNN x2 > EASU if still short > RCAS > grain  | `models.balanced` |
| `quality`     | deband > larger luma CNN x2 > EASU if still short > RCAS > grain | `models.quality`  |
| `auto`        | The strongest available preset that fits the frame budget        | -                 |

- EASU only runs when the screen is larger than the video; otherwise the frame gets the 1x
  cleanup (deband, RCAS, grain) and is scaled to the screen as it is drawn.
- The CNN only runs when the screen is more than 1.2x the video on both axes. Below that, a CNN
  preset runs the `performance` stages.
- A CNN preset without weights is not in the menu and auto never picks it. `setMode('quality')`
  without weights selects the strongest preset that can run, `performance` at least. If configured
  weights fail to load or validate, that preset is withdrawn with a console warning.

### Tuning

`tuning: { deband, sharpness, grain }`, each 0..1 (0 turns the stage off):

| Stage       | Default | Meaning                                                                        |
| ----------- | ------- | ------------------------------------------------------------------------------ |
| `deband`    | `0.5`   | Threshold 3/255 at 0.5 (6/255 at 1); search radius 16 px at 1080p, scaled      |
| `sharpness` | `0.4`   | RCAS strength: 1 is AMD's full strength, 0.5 one stop softer, 0.4 1.2 stops    |
| `grain`     | `0.25`  | Peak grain about 2.5/255, strongest in the midtones, fading to black and white |

The defaults are deliberately mild: halos on skin are the first thing a viewer notices.

### CNN weights

`balanced` and `quality` run a luma-only network: one channel in (Y'), a stack of 3x3
convolutions, and a depth-to-space to 2x, predicting a residual that is added to a bilinear 2x
upscale of the frame. Weights come as JSON:

```json
{
  "arch": "artcnn-like",
  "scale": 2,
  "channels": 16,
  "luma": "bt709",
  "layers": [
    { "type": "conv3x3", "in": 1, "out": 16, "relu": true, "weights": [], "bias": [] },
    { "type": "conv3x3", "in": 16, "out": 16, "relu": true, "weights": [], "bias": [] },
    { "type": "conv3x3", "in": 16, "out": 4, "relu": false, "weights": [], "bias": [] },
    { "type": "depth_to_space", "scale": 2 }
  ]
}
```

- `weights` is PyTorch's `Conv2d.weight` flattened row-major, `[out, in, 3, 3]`; `bias` is `[out]`.
- Convolutions pad by clamping to the edge (replicate, as trained; see training/README.md); `depth_to_space` is
  `nn.PixelShuffle`; the base is `F.interpolate(scale_factor=2, mode='bilinear', align_corners=False)`.
- Values are in 0..1. `luma` is `bt709` (default) or `bt601`, whichever the network was trained on.
- Scales 2, 3 and 4 are accepted; at most 64 channels per layer and 32 layers.

The package ships trained weights in `models/` (`balanced.json`, `quality.json`; preview weights,
see [`models/NOTICE.md`](models/NOTICE.md)). Serve them from your own origin and point the plugin
at them, or import them:

```ts
artplayerPluginLiveActionSR({
  models: {
    balanced: '/models/balanced.json', // a URL, fetched on first use
    quality: () =>
      import('artplayer-plugin-liveaction-sr/models/quality.json').then((m) => m.default), // or a loader, or the object itself
  },
});
```

On the GPU, activations live in `rgba16float` texture arrays and each convolution is one compute
dispatch; with `shader-f16` the arithmetic and weights are half precision. `parseModel` is exported
to validate a file ahead of time.

### How auto decides

1. Wait for the first decoded frame, and watch up to 8 frames of a playing video to learn its frame rate.
2. Run each available preset, weakest first, on that frame at the real resolution and the real on-screen size: 3 warm-up frames, then the median of 7.
3. Pick the strongest preset whose median GPU time is **at most the frame budget**: `frameBudgetMs`, or by default `min(12 ms, 35% of the frame interval)` - 12 ms for 24-30 fps film, 5.8 ms for 60 fps. Stop measuring at the first miss.
4. Cache the answer in `localStorage`, keyed by GPU, video resolution, screen size, budget and the set of available presets, for 30 days.

A runtime downgrade overwrites the cached answer, so a device that benchmarked optimistically does
not repeat the mistake.

### When it steps down

The plugin judges playback in windows of 90 frames. A frame is bad when it took longer than
`slowFrameMs` on the GPU - by default 60% of the frame interval within 16..33 ms (16 ms at 60 fps,
25 ms at 24 fps), updated as the frame rate is measured. Frames skipped because the previous one
was still in flight, and frames the browser reports as dropped, only count beyond 6 per window. A
window is bad when more than a quarter of it is bad.

Frames right after a (re)build, playback start, a seek, a stall or a tab switch drop on any GPU, so
those never count: the watch starts over on each of them, with 30 warm-up frames and 2 seconds in
which skipped and dropped frames are ignored.

What happens after **two bad windows in a row** depends on who chose the preset:

- **A preset the viewer picked** is never lowered or turned off. It keeps rendering, and ArtPlayer
  shows `labels.struggling` once per source; `getStats().struggling` turns `true`.
- **Auto** steps down one available preset and starts watching again. Its lowest preset,
  `performance`, only steps to `off` after **three** bad windows in a row.

With `autoDowngrade: false`, auto keeps its preset too and shows the same notice.

### Audio sync

The enhanced frame reaches the screen after the plain one would have, by the GPU time for that
frame. With `syncAudio` (the default), the plugin measures that lag, takes the median of the last
30 frames, and delays the video's audio by as much through Web Audio
(`video` -> `DelayNode` -> speakers), changing it at most 2% of real time so it never clicks. It is
capped at 250 ms. The audio is only rerouted once a frame has been processed (which proves the
video is readable) and after the page's first user gesture; a media element can be routed through
Web Audio only once, so if your page already does that, the plugin leaves the audio alone. Skipped
on iOS.

## Options

| Option            | Type                                       | Default                                        | Description                                                                      |
| ----------------- | ------------------------------------------ | ---------------------------------------------- | -------------------------------------------------------------------------------- |
| `mode`            | `Mode`                                     | `'auto'`                                       | Starting mode                                                                    |
| `models`          | `{ balanced?, quality? }`                  | none                                           | Weights per CNN preset: JSON object, URL, or a function resolving to one         |
| `tuning`          | `Partial<Tuning>`                          | `{ deband: 0.5, sharpness: 0.4, grain: 0.25 }` | Stage strengths, 0..1                                                            |
| `labels`          | `Partial<LiveActionSrLabels>`              | English                                        | Any of the user-facing strings (see below)                                       |
| `onModeChange`    | `(mode: Mode, active: ActiveMode) => void` | -                                              | Selection or active preset changed (menu, API, auto settling, downgrade)         |
| `onError`         | `(error: unknown) => void`                 | -                                              | Enhancing had to stop: cross-origin video, lost GPU device, build failure        |
| `setting`         | `boolean`                                  | `true`                                         | Add the entry to ArtPlayer's settings menu (needs `setting: true` on the player) |
| `icon`            | `string`                                   | film glyph                                     | SVG/HTML for the settings entry                                                  |
| `compare`         | `boolean`                                  | `false`                                        | Start in split view: original on the left, enhanced on the right                 |
| `frameBudgetMs`   | `number`                                   | from the frame rate                            | Auto's ceiling on median GPU time per frame                                      |
| `slowFrameMs`     | `number`                                   | from the frame rate                            | Frames slower than this count against the current preset                         |
| `autoDowngrade`   | `boolean`                                  | `true`                                         | Let auto step down when the GPU cannot keep up; a picked preset only warns       |
| `syncAudio`       | `boolean`                                  | `true`                                         | Delay the audio by as much as the enhanced picture trails                        |
| `maxOutputPixels` | `number`                                   | `3840 * 2160`                                  | Cap on the canvas backing store (the upscale target)                             |
| `cache`           | `boolean`                                  | `true`                                         | Remember auto's benchmark per device in `localStorage`                           |
| `cacheKey`        | `string`                                   | `'artplayer-plugin-liveaction-sr'`             | `localStorage` key prefix                                                        |
| `debug`           | `boolean`                                  | `false`                                        | Log frame rate, benchmark numbers, stages and downgrades to the console          |

## API

`art.plugins.artplayerPluginLiveActionSR`:

| Member                      | Description                                                                                                           |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `setMode(mode)`             | Select a mode; unknown values are ignored, a CNN preset without weights selects the best one that can run             |
| `getMode()`                 | The selected mode, `'auto'` included                                                                                  |
| `getActiveMode()`           | What is rendering now: auto's pick, auto's downgraded preset, or `'off'`                                              |
| `getAvailablePresets()`     | `['performance', ...]` plus every CNN preset with usable weights                                                      |
| `setCompare(on, position?)` | Split view on/off; `position` (0..1, default 0.5) is where the line sits                                              |
| `getCompare()`              | Whether the split view is on                                                                                          |
| `getStats()`                | `{ active, frameMs, native, target, upscaling, model, stages, fps, budgetMs, struggling, canvasLagMs, audioDelayMs }` |
| `supported`                 | WebGPU is usable (`false` until the check finishes)                                                                   |
| `ready`                     | `Promise<boolean>` resolving with `supported`; never rejects                                                          |
| `destroy()`                 | Remove the canvas and settings entry, free the GPU; also runs on `art.destroy()`                                      |

## Translating

```ts
artplayerPluginLiveActionSR({
  labels: {
    setting: 'Verbessern',
    auto: 'Automatisch',
    off: 'Aus',
    performance: 'Leistung',
    balanced: 'Ausgewogen',
    quality: 'Qualität',
    autoActive: 'Automatisch ({mode})', // {mode} becomes the label of the preset auto chose
    unsupported: 'Nicht unterstützt',
    struggling: 'Für diese GPU zu aufwendig; wähle eine leichtere Stufe',
    blocked: 'Dieses Video kann nicht verbessert werden (CORS)',
  },
});
```

## Same-origin and CORS

WebGPU copies each video frame into a texture, and browsers refuse to let a page read pixels from a
**cross-origin video that was not loaded with CORS**. For enhancing to work the video must either:

- come from the **same origin** as the page (for example through your own streaming proxy), or
- be served with `Access-Control-Allow-Origin` **and** loaded with `crossorigin="anonymous"`
  (`moreVideoAttr: { crossOrigin: 'anonymous' }` in ArtPlayer). With hls.js or dash.js the segments
  are fetched by script, so the playlist and every segment need the header.

A tainted video does not break playback: the first frame copy throws a `SecurityError`, the plugin
logs one warning, shows `labels.blocked` as an ArtPlayer notice, calls `onError`, switches that
source to `off` and tries again on the next source. DRM-protected video can never be read.

## Browser support

WebGPU is required: see [caniuse.com/webgpu](https://caniuse.com/webgpu) - on by default in
Chromium on Windows, macOS, ChromeOS and Android 12+, Firefox on Windows (141+) and macOS, and
Safari 26; not yet in Firefox on Linux and Android. Where it is missing the settings entry is not
added and `supported` is `false`. `shader-f16` is used when the adapter offers it.
`requestVideoFrameCallback` drives the loop and the frame-rate measurement, with a
`requestAnimationFrame` fallback (and the 30 fps budget). Browsers with native video
super-resolution (RTX VSR, Edge VSR) apply it to the `<video>` only; there `off` is a real choice.

## Get started (development)

Prerequisites: **Bun** (or Node 22+).

```bash
bun install
bun run dev        # demo page with ArtPlayer + hls.js -> http://localhost:5173/demo/
bun run test       # vitest: modes, frame rate, pass planning, weights packing, monitor, options, cache, layout
bun run build      # dist/index.js (ESM) + .d.ts, dist/artplayer-plugin-liveaction-sr.js (browser global)
```

The demo plays Tears of Steel (CC BY 3.0) and takes any mp4 or m3u8 URL (`?url=...&mode=...`).
`?model=test` plugs random stand-in weights into `balanced` and `quality` to run and time the CNN
path; `?model=<url>` loads a real weights file. `/demo/compare.html` is a side-by-side view: open a
file from disk (or drop it on the player) and drag the split line.

The CNN training pipeline lives in [`training/`](training/) and is not part of the npm package.

## Licence

The plugin's code is [MIT](LICENSE). The FSR 1 port (`src/shaders/fsr1.wgsl.ts`) is derived from
AMD FidelityFX Super Resolution 1.0, MIT, Copyright (c) 2021 Advanced Micro Devices, Inc.; its
notice is kept in the source and in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), which ships
with the package. The deband, grain and CNN shaders were written for this project; no LGPL/GPL
code (libplacebo, mpv, RAVU, FSRCNNX, Magpie) is included. Model weights are not bundled and carry
their own licence.

## Credits

[AMD FidelityFX FSR 1](https://github.com/GPUOpen-Effects/FidelityFX-FSR) ·
[ArtPlayer](https://github.com/zhw2590582/ArtPlayer) by Harvey Zhao ·
the f3kdb debanding idea · the [research summary](docs/research.md) behind the design. The settings
icon is Lucide's `film` (ISC).
