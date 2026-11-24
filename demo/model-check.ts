import { acquireGpu, type GpuContext } from '../src/gpu';
import { graphInputReference, isGraphModel, runGraphReference } from '../src/graph';
import { fromHalfBits, loadModel, toHalfBits, type ModelWeights } from '../src/model';
import type { Preset } from '../src/modes';
import { FramePipeline, type GraphKernelOptions } from '../src/pipeline';
import type { ActivationFormat } from '../src/shaders/graph.wgsl';
import { DEFAULT_TUNING } from '../src/plan';

// Runs a CNN preset's real GPU pipeline on one still frame and compares the result with a CPU
// reference: the plugin's own (`runGraphReference` / `runModelReference`) and, when present, a
// float32 dump from tools/efrlfn/convert.py --reference (PyTorch-equivalent numpy). Then times
// presets at video sizes. Parameters:
//   ?model=/models/ultra.json  weights to check (default ultra)
//   ?frame=/tmp/frame.png      a small frame (the CPU reference runs the whole network in JS)
//   ?ref=/tmp/frame.ref.f32    optional [3, 2H, 2W] float32 reference output
//   ?f16=0                     force the f32 path
//   ?bench=quality,ultra       presets to time (weights from /models/<preset>.json); bench=0 skips
//   ?sizes=1280x720,1920x1080  source sizes to time at, each upscaled 2x
//   ?kernels=1                 also time graph convolution variants (?variants=2x8,4x4h)

const params = new URLSearchParams(location.search);
const log = document.getElementById('log') as HTMLPreElement;
const results: Record<string, unknown> = {};
(window as unknown as { modelCheck: unknown }).modelCheck = results;

function print(line: string): void {
  log.textContent += `${line}\n`;
}

async function readTexture(gpu: GpuContext, texture: GPUTexture): Promise<Float32Array> {
  const { width, height } = texture;
  const bytesPerRow = Math.ceil((width * 8) / 256) * 256;
  const buffer = gpu.device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = gpu.device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow }, [width, height]);
  gpu.device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const halves = new Uint16Array(buffer.getMappedRange());
  const rgba = new Float32Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width * 4; x++) {
      rgba[y * width * 4 + x] = fromHalfBits(halves[(y * bytesPerRow) / 2 + x]);
    }
  }
  buffer.destroy();
  return rgba;
}

/** Runs one frame through the preset with only the CNN stage (no deband, no sharpening). */
async function runOnFrame(
  gpu: GpuContext,
  model: ModelWeights,
  preset: Preset,
  bitmap: ImageBitmap,
  f16: boolean,
  graphKernel?: Partial<GraphKernelOptions>,
): Promise<Float32Array> {
  const native = { width: bitmap.width, height: bitmap.height };
  const target = { width: native.width * model.scale, height: native.height * model.scale };
  const chain = await FramePipeline.create(gpu.device, {
    preset,
    native,
    target,
    tuning: { deband: 0, sharpness: 0, grain: 0 },
    model,
    f16,
    graphKernel,
  });
  try {
    const stages = chain.plan.stages.map((s) => s.kind).join(' > ');
    if (stages !== 'model') throw new Error(`expected only the model stage, got ${stages}`);
    gpu.device.queue.copyExternalImageToTexture({ source: bitmap }, { texture: chain.input }, [
      native.width,
      native.height,
    ]);
    const encoder = gpu.device.createCommandEncoder();
    chain.encode(encoder);
    gpu.device.queue.submit([encoder.finish()]);
    return await readTexture(gpu, chain.output);
  } finally {
    chain.destroy();
  }
}

function framePlanes(bitmap: ImageBitmap): [Float32Array, Float32Array, Float32Array] {
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const context = canvas.getContext('2d') as OffscreenCanvasRenderingContext2D;
  context.drawImage(bitmap, 0, 0);
  const { data } = context.getImageData(0, 0, bitmap.width, bitmap.height);
  const area = bitmap.width * bitmap.height;
  const planes: [Float32Array, Float32Array, Float32Array] = [
    new Float32Array(area),
    new Float32Array(area),
    new Float32Array(area),
  ];
  for (let i = 0; i < area; i++) {
    for (let c = 0; c < 3; c++) planes[c][i] = data[i * 4 + c] / 255;
  }
  return planes;
}

/** Compares the GPU's RGBA output with [3, H, W] reference planes, clamped to 0..1 like the GPU. */
function compare(gpu: Float32Array, reference: Float32Array[], width: number, height: number) {
  let worst = 0;
  let sumSq = 0;
  let worstAt = '';
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < width * height; i++) {
      const expected = Math.min(1, Math.max(0, reference[c][i]));
      const diff = Math.abs(gpu[i * 4 + c] - expected);
      sumSq += diff * diff;
      if (diff > worst) {
        worst = diff;
        worstAt = `c${c} x${i % width} y${Math.floor(i / width)}`;
      }
    }
  }
  const mse = sumSq / (3 * width * height);
  return { maxAbs: worst, maxAbs255: worst * 255, psnr: 10 * Math.log10(1 / mse), worstAt };
}

async function time(
  gpu: GpuContext,
  model: ModelWeights | null,
  preset: Preset,
  native: { width: number; height: number },
  full: boolean,
  graphKernel?: Partial<GraphKernelOptions>,
  frames = 12,
): Promise<number> {
  const target = { width: native.width * 2, height: native.height * 2 };
  const chain = await FramePipeline.create(gpu.device, {
    preset,
    native,
    target,
    tuning: full ? DEFAULT_TUNING : { deband: 0, sharpness: 0, grain: 0 },
    model,
    f16: gpu.f16,
    graphKernel,
  });
  const times: number[] = [];
  try {
    for (let i = 0; i < frames + 3; i++) {
      const started = performance.now();
      const encoder = gpu.device.createCommandEncoder();
      chain.encode(encoder);
      gpu.device.queue.submit([encoder.finish()]);
      await gpu.device.queue.onSubmittedWorkDone();
      if (i >= 3) times.push(performance.now() - started);
    }
  } finally {
    chain.destroy();
  }
  times.sort((a, b) => a - b);
  return times[times.length >> 1];
}

async function main(): Promise<void> {
  const gpu = await acquireGpu();
  if (!gpu) {
    print('no WebGPU');
    return;
  }
  let validationErrors = 0;
  gpu.device.addEventListener('uncapturederror', (event) => {
    validationErrors++;
    print(`GPU ERROR: ${(event as GPUUncapturedErrorEvent).error.message}`);
  });
  gpu.device.pushErrorScope('validation');
  print(`adapter: ${gpu.name}, shader-f16: ${gpu.f16}`);
  results.adapter = gpu.name;

  const modelUrl = params.get('model') ?? '/models/ultra.json';
  const model = await loadModel(modelUrl);
  print(`model ${modelUrl}: ${model.arch}, x${model.scale}`);

  const frameUrl = params.get('frame') ?? '/tmp/frame.png';
  const bitmap = await createImageBitmap(await (await fetch(frameUrl)).blob(), {
    colorSpaceConversion: 'none',
  });
  const { width, height } = bitmap;
  const outW = width * model.scale;
  const outH = height * model.scale;
  print(`frame ${frameUrl}: ${width}x${height} -> ${outW}x${outH}`);

  const started = performance.now();
  const rgb = framePlanes(bitmap);
  // `?inputf16=1`: round the CPU input the way the plugin's rgba16float input texture holds it.
  if (params.get('inputf16') === '1') {
    for (const plane of rgb) plane.forEach((v, i) => (plane[i] = fromHalfBits(toHalfBits(v))));
  }
  let cpu: Float32Array[];
  if (isGraphModel(model)) {
    cpu = runGraphReference(model, graphInputReference(model, rgb), width, height);
  } else {
    throw new Error('the check compares RGB outputs; use a graph model');
  }
  print(`CPU reference (TypeScript) in ${((performance.now() - started) / 1000).toFixed(1)} s`);

  let pytorch: Float32Array[] | null = null;
  const refUrl = params.get('ref') ?? (params.has('frame') ? null : '/tmp/frame.ref.f32');
  if (refUrl) {
    const response = await fetch(refUrl);
    if (response.ok) {
      const all = new Float32Array(await response.arrayBuffer());
      if (all.length === 3 * outW * outH) {
        pytorch = [0, 1, 2].map((c) => all.subarray(c * outW * outH, (c + 1) * outW * outH));
        results.cpuVsPytorch = compare(
          Float32Array.from({ length: outW * outH * 4 }, (_, i) =>
            i % 4 === 3 ? 1 : Math.min(1, Math.max(0, cpu[i % 4][i >> 2])),
          ),
          pytorch,
          outW,
          outH,
        );
        print(`CPU reference vs PyTorch: ${JSON.stringify(results.cpuVsPytorch)}`);
      }
    }
  }

  const variants: [boolean, ActivationFormat][] = [];
  for (const f16 of gpu.f16 && params.get('f16') !== '0' ? [false, true] : [false]) {
    for (const format of ['rgba16float', 'rgba32float'] as const) variants.push([f16, format]);
  }
  for (const [f16, format] of variants) {
    const out = await runOnFrame(gpu, model, 'ultra', bitmap, f16, { format });
    const key = `${f16 ? 'f16' : 'f32'} math, ${format}`;
    const vsCpu = compare(out, cpu, outW, outH);
    results[`gpu_${key}_vs_cpu`] = vsCpu;
    print(`GPU ${key} vs CPU reference: ${JSON.stringify(vsCpu)}`);
    if (pytorch) {
      const vsTorch = compare(out, pytorch, outW, outH);
      results[`gpu_${key}_vs_pytorch`] = vsTorch;
      print(`GPU ${key} vs PyTorch:       ${JSON.stringify(vsTorch)}`);
    }
    // A small region, printed, to eyeball.
    const row = 40;
    const sample = (planes: (i: number) => number) =>
      Array.from({ length: 4 }, (_, k) => planes(row * outW + 60 + k).toFixed(4)).join(' ');
    print(`  R at y${row} x60..63  gpu ${sample((i) => out[i * 4])}`);
    print(`  R at y${row} x60..63  cpu ${sample((i) => Math.min(1, Math.max(0, cpu[0][i])))}`);
  }

  const scopeError = await gpu.device.popErrorScope();
  if (scopeError) {
    validationErrors++;
    print(`VALIDATION: ${scopeError.message}`);
  }

  const bench = params.get('bench') ?? 'quality,ultra';
  if (bench !== '0') {
    const sizes = (params.get('sizes') ?? '1280x720,1920x1080').split(',').map((s) => {
      const [w, h] = s.split('x').map(Number);
      return { width: w, height: h };
    });
    const presets = bench.split(',') as Preset[];
    const models = new Map<Preset, ModelWeights | null>();
    for (const preset of presets) {
      models.set(
        preset,
        preset === 'performance' ? null : await loadModel(`/models/${preset}.json`),
      );
    }
    const timing: Record<string, number> = {};
    for (const size of sizes) {
      for (const preset of presets) {
        for (const full of [false, true]) {
          const ms = await time(gpu, models.get(preset)!, preset, size, full);
          const key = `${preset} ${size.width}x${size.height}->${size.width * 2}x${size.height * 2} ${full ? 'full chain' : 'CNN only'}`;
          timing[key] = Math.round(ms * 100) / 100;
          print(`${key}: ${ms.toFixed(2)} ms`);
        }
      }
    }
    results.timing = timing;
    if (params.get('kernels') === '1' && isGraphModel(model)) {
      const variants: Record<string, number> = {};
      // `?variants=2x8,4x4h`: output groups x pixels per invocation, h = accumulate in f16.
      const size = sizes[0];
      const list = (params.get('variants') ?? '2x4,2x8,4x4,4x8,2x8h,4x4h').split(',');
      for (const variant of list) {
        const [perInvocation, pixels] = variant.replace('h', '').split('x').map(Number);
        const accumulateF32 = !variant.endsWith('h');
        const ms = await time(gpu, model, 'ultra', size, false, {
          perInvocation,
          pixels,
          accumulateF32,
        });
        variants[variant] = Math.round(ms * 100) / 100;
        print(`kernel ${variant} at ${size.width}x${size.height}: ${ms.toFixed(2)} ms`);
      }
      results.kernels = variants;
    }
  }
  results.validationErrors = validationErrors;
  print(`validation errors: ${validationErrors}`);
  results.done = true;
}

main().catch((error: unknown) => {
  print(`FAILED: ${error instanceof Error ? (error.stack ?? error.message) : String(error)}`);
  results.error = String(error);
  results.done = true;
});
