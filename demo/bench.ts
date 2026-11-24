import { loadModel, toHalfBits, type ModelWeights } from '../src/model';
import type { Preset } from '../src/modes';
import { FramePipeline } from '../src/pipeline';
import { DEFAULT_TUNING, type Tuning } from '../src/plan';

// GPU time per frame of the real preset pipelines, from GPU timestamp queries (a readback-
// synchronized loop when the adapter has no `timestamp-query`). Each frame is one compute pass;
// the frames are submitted back to back after a warm-up, and the median is reported. Parameters:
//   ?presets=performance,balanced,quality,ultra  weights from /models/<preset>.json
//   ?sizes=640x360,1280x720,1280x720@3840x2160   source sizes, upscaled 2x unless @target is given
//   ?stages=cnn,full     cnn: the CNN stage alone (no deband, no RCAS); full: the default tuning
//   ?precision=f16,f32   f16 needs `shader-f16`
//   ?frames=40&warmup=10
// The full chain is what FramePipeline runs: deband > CNN (> EASU) > RCAS. The present pass
// (scaling to the canvas, grain) is not included.

const params = new URLSearchParams(location.search);
const log = document.getElementById('log') as HTMLPreElement;
const results: Record<string, unknown> = {};
(window as unknown as { bench: unknown }).bench = results;

function print(line: string): void {
  log.textContent += `${line}\n`;
}

function list(name: string, fallback: string): string[] {
  return (params.get(name) ?? fallback).split(',').filter((s) => s !== '');
}

interface Job {
  native: { width: number; height: number };
  target: { width: number; height: number };
}

function parseSize(text: string): Job {
  const [source, target] = text.split('@');
  const [width, height] = source.split('x').map(Number);
  const [tw, th] = target ? target.split('x').map(Number) : [width * 2, height * 2];
  return { native: { width, height }, target: { width: tw, height: th } };
}

/** Random half-float RGBA, so nothing is constant-folded by content. */
function fillInput(device: GPUDevice, texture: GPUTexture): void {
  const { width, height } = texture;
  const data = new Uint16Array(width * height * 4);
  let seed = 12345;
  for (let i = 0; i < data.length; i++) {
    seed = (seed * 1103515245 + 12345) >>> 0;
    data[i] = toHalfBits((seed >>> 8) / 2 ** 24);
  }
  device.queue.writeTexture({ texture }, data, { bytesPerRow: width * 8 }, [width, height]);
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

async function measure(
  device: GPUDevice,
  timestamps: boolean,
  preset: Preset,
  model: ModelWeights | null,
  job: Job,
  tuning: Tuning,
  f16: boolean,
  frames: number,
  warmup: number,
): Promise<{ ms: number; p10: number; p90: number; stages: string }> {
  const chain = await FramePipeline.create(device, {
    preset,
    native: job.native,
    target: job.target,
    tuning,
    model,
    f16,
  });
  try {
    fillInput(device, chain.input);
    for (let i = 0; i < warmup; i++) {
      const encoder = device.createCommandEncoder();
      chain.encode(encoder);
      device.queue.submit([encoder.finish()]);
    }
    await device.queue.onSubmittedWorkDone();
    const times: number[] = [];
    if (timestamps) {
      const querySet = device.createQuerySet({ type: 'timestamp', count: frames * 2 });
      const resolve = device.createBuffer({
        size: frames * 16,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      const read = device.createBuffer({
        size: frames * 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      for (let i = 0; i < frames; i++) {
        const encoder = device.createCommandEncoder();
        chain.encode(encoder, {
          querySet,
          beginningOfPassWriteIndex: i * 2,
          endOfPassWriteIndex: i * 2 + 1,
        });
        device.queue.submit([encoder.finish()]);
      }
      const encoder = device.createCommandEncoder();
      encoder.resolveQuerySet(querySet, 0, frames * 2, resolve, 0);
      encoder.copyBufferToBuffer(resolve, 0, read, 0, frames * 16);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const ns = new BigInt64Array(read.getMappedRange());
      for (let i = 0; i < frames; i++) times.push(Number(ns[i * 2 + 1] - ns[i * 2]) / 1e6);
      read.unmap();
      querySet.destroy();
      resolve.destroy();
      read.destroy();
    } else {
      for (let i = 0; i < frames; i++) {
        const started = performance.now();
        const encoder = device.createCommandEncoder();
        chain.encode(encoder);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        times.push(performance.now() - started);
      }
    }
    const sorted = [...times].sort((a, b) => a - b);
    return {
      ms: median(times),
      p10: sorted[Math.floor(frames * 0.1)],
      p90: sorted[Math.floor(frames * 0.9)],
      stages: chain.plan.stages.map((s) => s.kind).join(' > '),
    };
  } finally {
    chain.destroy();
  }
}

async function main(): Promise<void> {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    print('no WebGPU');
    results.done = true;
    return;
  }
  const timestamps = adapter.features.has('timestamp-query');
  const hasF16 = adapter.features.has('shader-f16');
  const device = await adapter.requestDevice({
    requiredFeatures: [
      ...(timestamps ? (['timestamp-query'] as const) : []),
      ...(hasF16 ? (['shader-f16'] as const) : []),
    ],
  });
  let validationErrors = 0;
  device.addEventListener('uncapturederror', (event) => {
    validationErrors++;
    print(`GPU ERROR: ${(event as GPUUncapturedErrorEvent).error.message}`);
  });
  const info = adapter.info;
  results.adapter = `${info.vendor} ${info.architecture} ${info.description}`.trim();
  results.userAgent = navigator.userAgent;
  results.method = timestamps ? 'timestamp-query' : 'readback loop';
  print(`adapter: ${String(results.adapter)}, shader-f16: ${hasF16}, timing: ${results.method}`);

  const presets = list('presets', 'performance,balanced,quality,ultra') as Preset[];
  const jobs = list('sizes', '640x360,1280x720,1920x1080,1280x720@3840x2160').map(parseSize);
  const stages = list('stages', 'cnn,full');
  const precisions = list('precision', hasF16 ? 'f16,f32' : 'f32').filter(
    (p) => p === 'f32' || hasF16,
  );
  const frames = Number(params.get('frames') ?? 40);
  const warmup = Number(params.get('warmup') ?? 10);

  const models = new Map<Preset, ModelWeights | null>();
  for (const preset of presets) {
    models.set(preset, preset === 'performance' ? null : await loadModel(`/models/${preset}.json`));
  }

  const timing: Record<string, unknown>[] = [];
  results.timing = timing;
  for (const job of jobs) {
    for (const preset of presets) {
      for (const stage of stages) {
        // `performance` has no CNN; its "cnn" row would only be EASU.
        if (stage === 'cnn' && preset === 'performance') continue;
        for (const precision of preset === 'performance' ? ['f32'] : precisions) {
          const tuning = stage === 'full' ? DEFAULT_TUNING : { deband: 0, sharpness: 0, grain: 0 };
          const r = await measure(
            device,
            timestamps,
            preset,
            models.get(preset)!,
            job,
            tuning,
            precision === 'f16',
            frames,
            warmup,
          );
          const row = {
            preset,
            source: `${job.native.width}x${job.native.height}`,
            target: `${job.target.width}x${job.target.height}`,
            stage,
            precision,
            ms: Math.round(r.ms * 100) / 100,
            p10: Math.round(r.p10 * 100) / 100,
            p90: Math.round(r.p90 * 100) / 100,
            stages: r.stages,
          };
          timing.push(row);
          print(
            `${row.preset.padEnd(11)} ${row.source}->${row.target} ${stage.padEnd(4)} ${precision}: ` +
              `${r.ms.toFixed(2)} ms (p10 ${r.p10.toFixed(2)}, p90 ${r.p90.toFixed(2)}) [${r.stages}]`,
          );
        }
      }
    }
  }
  await device.queue.onSubmittedWorkDone();
  results.validationErrors = validationErrors;
  print(`validation errors: ${validationErrors}`);
  results.done = true;
}

main().catch((error: unknown) => {
  print(`FAILED: ${error instanceof Error ? (error.stack ?? error.message) : String(error)}`);
  results.error = String(error);
  results.done = true;
});
