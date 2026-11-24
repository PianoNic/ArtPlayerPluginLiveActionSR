import type { Size } from './layout.js';
import type { ModelWeights } from './model.js';
import type { Preset } from './modes.js';
import { grainAmplitude, type PassPlan, type Tuning } from './plan.js';
import { FramePipeline } from './pipeline.js';
import { PRESENT_WGSL } from './shaders/present.wgsl.js';

export interface GpuContext {
  device: GPUDevice;
  format: GPUTextureFormat;
  /** Identifies the adapter for the benchmark cache: vendor, architecture, device. */
  name: string;
  /** The device has `shader-f16`; the CNN then runs in half precision. */
  f16: boolean;
}

/** A WebGPU device, or null when WebGPU is missing, blocklisted or fails. Never throws. */
export async function acquireGpu(): Promise<GpuContext | null> {
  try {
    const gpu = typeof navigator !== 'undefined' ? navigator.gpu : undefined;
    if (!gpu) return null;
    const adapter = await gpu.requestAdapter();
    if (!adapter) return null;
    const f16 = adapter.features.has('shader-f16');
    const device = await adapter.requestDevice({
      requiredFeatures: f16 ? ['shader-f16'] : [],
    });
    const info = adapter.info;
    const name =
      [info?.vendor, info?.architecture, info?.device, info?.description]
        .filter((part) => typeof part === 'string' && part !== '')
        .join('|') || 'unknown';
    return { device, format: gpu.getPreferredCanvasFormat(), name, f16 };
  } catch {
    return null;
  }
}

/** What one build needs besides the sizes. */
export interface ChainOptions {
  tuning: Tuning;
  model: ModelWeights | null;
}

function buildChain(
  gpu: GpuContext,
  preset: Preset,
  native: Size,
  target: Size,
  options: ChainOptions,
): Promise<FramePipeline> {
  return FramePipeline.create(gpu.device, {
    preset,
    native,
    target,
    tuning: options.tuning,
    model: options.model,
    f16: gpu.f16,
  });
}

/** Whether a colour space carries an HDR transfer function (PQ / HLG). */
export function isHdrTransfer(transfer: string | null | undefined): boolean {
  return transfer === 'pq' || transfer === 'hlg';
}

/**
 * HDR sources (PQ / HLG, usually BT.2020) need the browser's own tone mapping. The <video> element
 * tone-maps them for an SDR screen, and so does drawing them onto a 2D canvas, but
 * `copyExternalImageToTexture` straight from the video does not: it hands over the raw signal,
 * which looks washed out and far too bright (measured on a PQ stream: mean luma 19.6 on the video,
 * 111 after the copy). For those, the frame goes through a 2D canvas first, so the upscaled picture
 * matches the plain one. SDR video takes the direct copy.
 */
class FrameSource {
  private key = '';
  private hdr = false;
  private staging: OffscreenCanvas | null = null;
  private context: OffscreenCanvasRenderingContext2D | null = null;

  /** What to copy from for the current frame of `video`. */
  for(video: HTMLVideoElement): HTMLVideoElement | OffscreenCanvas {
    const key = `${video.currentSrc}|${video.videoWidth}x${video.videoHeight}`;
    if (key !== this.key) {
      this.key = key;
      this.hdr = detectHdr(video);
    }
    if (!this.hdr || typeof OffscreenCanvas === 'undefined') return video;
    const { videoWidth: w, videoHeight: h } = video;
    if (!this.staging || this.staging.width !== w || this.staging.height !== h) {
      this.staging = new OffscreenCanvas(w, h);
      this.context = this.staging.getContext('2d', { alpha: false });
    }
    if (!this.context) return video;
    this.context.drawImage(video, 0, 0, w, h);
    return this.staging;
  }
}

function detectHdr(video: HTMLVideoElement): boolean {
  if (typeof VideoFrame === 'undefined') return false;
  let frame: VideoFrame | null = null;
  try {
    frame = new VideoFrame(video);
    return isHdrTransfer(frame.colorSpace?.transfer);
  } catch {
    return false; // no frame yet, or not readable: the direct copy reports that itself
  } finally {
    frame?.close();
  }
}

/**
 * Copies the current frame in and records the chain. Throws a `SecurityError` for a cross-origin
 * video that was not loaded with CORS: WebGPU may not read its pixels.
 */
function encodeChain(
  gpu: GpuContext,
  chain: FramePipeline,
  video: HTMLVideoElement,
  source: FrameSource,
): GPUCommandEncoder {
  gpu.device.queue.copyExternalImageToTexture(
    { source: source.for(video) },
    { texture: chain.input },
    [chain.input.width, chain.input.height],
  );
  const encoder = gpu.device.createCommandEncoder();
  chain.encode(encoder);
  return encoder;
}

/** Owns the canvas context and the one chain currently on screen. */
export class Renderer {
  private readonly context: GPUCanvasContext;
  private readonly sampler: GPUSampler;
  private readonly params: GPUBuffer;
  private readonly renderPipeline: Promise<GPURenderPipeline>;
  private chain: FramePipeline | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private grain = 0;
  private frame = 0;
  private readonly source = new FrameSource();

  constructor(
    private readonly gpu: GpuContext,
    canvas: HTMLCanvasElement,
  ) {
    const context = canvas.getContext('webgpu');
    if (!context) throw new Error('Could not get a WebGPU canvas context');
    this.context = context;
    context.configure({ device: gpu.device, format: gpu.format, alphaMode: 'opaque' });
    this.sampler = gpu.device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.params = gpu.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const module = gpu.device.createShaderModule({ code: PRESENT_WGSL });
    this.renderPipeline = gpu.device.createRenderPipelineAsync({
      layout: 'auto',
      vertex: { module, entryPoint: 'vert_main' },
      fragment: { module, entryPoint: 'frag_main', targets: [{ format: gpu.format }] },
      primitive: { topology: 'triangle-list' },
    });
  }

  /** Replaces the chain on screen. The previous one's textures are freed immediately. */
  async build(
    preset: Preset,
    native: Size,
    target: Size,
    options: ChainOptions,
  ): Promise<PassPlan> {
    const chain = await buildChain(this.gpu, preset, native, target, options);
    let renderPipeline: GPURenderPipeline;
    try {
      renderPipeline = await this.renderPipeline;
    } catch (error) {
      chain.destroy();
      throw error;
    }
    this.release();
    this.chain = chain;
    this.grain = grainAmplitude(options.tuning.grain);
    this.bindGroup = this.gpu.device.createBindGroup({
      layout: renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: chain.output.createView() },
        { binding: 2, resource: { buffer: this.params } },
      ],
    });
    return chain.plan;
  }

  get ready(): boolean {
    return this.chain !== null;
  }

  /** Renders one frame and resolves with the milliseconds until the GPU finished it. */
  async render(video: HTMLVideoElement): Promise<number> {
    const chain = this.chain;
    const bindGroup = this.bindGroup;
    if (!chain || !bindGroup) return 0;
    const renderPipeline = await this.renderPipeline;
    if (chain !== this.chain) return 0; // rebuilt meanwhile
    const started = performance.now();
    const encoder = encodeChain(this.gpu, chain, video, this.source);
    const params = new ArrayBuffer(16);
    new Float32Array(params, 0, 1)[0] = this.grain;
    new Uint32Array(params, 4, 1)[0] = this.frame++ >>> 0;
    this.gpu.device.queue.writeBuffer(this.params, 0, params);
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    pass.setPipeline(renderPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    this.gpu.device.queue.submit([encoder.finish()]);
    await this.gpu.device.queue.onSubmittedWorkDone();
    return performance.now() - started;
  }

  release(): void {
    this.chain?.destroy();
    this.chain = null;
    this.bindGroup = null;
  }

  destroy(): void {
    this.release();
    try {
      this.params.destroy();
      this.context.unconfigure();
    } catch {
      // Device already gone.
    }
  }
}

/** GPU time of a preset on the current frame, per frame, after a few warm-up frames. Draws nothing. */
export async function measurePreset(
  gpu: GpuContext,
  video: HTMLVideoElement,
  preset: Preset,
  native: Size,
  target: Size,
  options: ChainOptions,
  { warmup = 3, frames = 7, timeoutMs = 2000 } = {},
): Promise<number[]> {
  const chain = await buildChain(gpu, preset, native, target, options);
  const times: number[] = [];
  const source = new FrameSource(); // the benchmark pays for the HDR canvas copy too
  try {
    const deadline = performance.now() + timeoutMs;
    for (let i = 0; i < warmup + frames; i++) {
      const started = performance.now();
      const encoder = encodeChain(gpu, chain, video, source);
      gpu.device.queue.submit([encoder.finish()]);
      await gpu.device.queue.onSubmittedWorkDone();
      if (i >= warmup) times.push(performance.now() - started);
      if (performance.now() > deadline) break;
    }
  } finally {
    chain.destroy();
  }
  return times;
}
