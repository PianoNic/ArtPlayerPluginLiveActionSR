import type { Size } from './layout.js';
import {
  convLayers,
  groups,
  packBias,
  packConvWeights,
  toHalfArray,
  type ModelWeights,
} from './model.js';
import type { Preset } from './modes.js';
import {
  debandRadius,
  debandThreshold,
  planPasses,
  rcasSharpness,
  type PassPlan,
  type Tuning,
} from './plan.js';
import { DEBAND_WGSL } from './shaders/deband.wgsl.js';
import { EASU_WGSL, RCAS_WGSL } from './shaders/fsr1.wgsl.js';
import { convWgsl, lumaWgsl, reconstructWgsl } from './shaders/model.wgsl.js';

export interface PipelineConfig {
  preset: Preset;
  native: Size;
  target: Size;
  tuning: Tuning;
  /** Weights for `balanced` / `quality`; null runs the `performance` stages. */
  model: ModelWeights | null;
  /** Run the CNN in half precision (`shader-f16`). */
  f16: boolean;
}

interface Pass {
  label: string;
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  workgroups: [number, number, number];
}

const FORMAT: GPUTextureFormat = 'rgba16float';
const WORKGROUP = 8;

// Compiling a pipeline takes far longer than building the textures: one per shader and device.
const pipelines = new WeakMap<GPUDevice, Map<string, Promise<GPUComputePipeline>>>();

function computePipeline(device: GPUDevice, code: string): Promise<GPUComputePipeline> {
  let cache = pipelines.get(device);
  if (!cache) {
    cache = new Map();
    pipelines.set(device, cache);
  }
  let pipeline = cache.get(code);
  if (!pipeline) {
    pipeline = device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
    });
    const map = cache;
    pipeline.catch(() => map.delete(code));
    cache.set(code, pipeline);
  }
  return pipeline;
}

function workgroupsFor(size: Size, z = 1): [number, number, number] {
  return [Math.ceil(size.width / WORKGROUP), Math.ceil(size.height / WORKGROUP), z];
}

const arrayView: GPUTextureViewDescriptor = { dimension: '2d-array' };

/**
 * The GPU side of one preset at one size: every intermediate texture, uniform and weight buffer,
 * and the compute passes in order. The frame goes into `input`, the result comes out of `output`.
 */
export class FramePipeline {
  private constructor(
    readonly input: GPUTexture,
    readonly output: GPUTexture,
    readonly plan: PassPlan,
    private readonly passes: Pass[],
    private readonly owned: { destroy(): void }[],
  ) {}

  static async create(device: GPUDevice, config: PipelineConfig): Promise<FramePipeline> {
    const { native, target, tuning, model } = config;
    const plan = planPasses(config.preset, native, target, tuning, model?.scale ?? null);
    const owned: { destroy(): void }[] = [];
    const storage =
      GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC;

    const texture = (size: Size, usage: number, layers = 1): GPUTexture => {
      const created = device.createTexture({
        size: [size.width, size.height, layers],
        format: FORMAT,
        usage,
      });
      owned.push(created);
      return created;
    };
    const buffer = (data: ArrayBufferView<ArrayBuffer>, usage: number): GPUBuffer => {
      const created = device.createBuffer({
        size: Math.max(16, Math.ceil(data.byteLength / 4) * 4),
        usage: usage | GPUBufferUsage.COPY_DST,
      });
      owned.push(created);
      device.queue.writeBuffer(created, 0, data.buffer, data.byteOffset, data.byteLength);
      return created;
    };

    try {
      const input = texture(
        native,
        GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
      );
      const passes: Pass[] = [];
      let current = input;

      for (const stage of plan.stages) {
        switch (stage.kind) {
          case 'deband': {
            const out = texture(stage.size, storage);
            const params = new ArrayBuffer(16);
            new Float32Array(params, 0, 2).set([
              debandThreshold(tuning.deband),
              debandRadius(native),
            ]);
            new Uint32Array(params, 8, 1)[0] = 2;
            const pipeline = await computePipeline(device, DEBAND_WGSL);
            passes.push({
              label: 'deband',
              pipeline,
              workgroups: workgroupsFor(stage.size),
              bindGroup: device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                  { binding: 0, resource: current.createView() },
                  { binding: 1, resource: out.createView() },
                  {
                    binding: 2,
                    resource: { buffer: buffer(new Uint8Array(params), GPUBufferUsage.UNIFORM) },
                  },
                ],
              }),
            });
            current = out;
            break;
          }

          case 'model': {
            if (!model) throw new Error('model stage without weights');
            const network = await modelPasses(device, model, config.f16, current, stage.size, {
              texture,
              buffer,
            });
            passes.push(...network.passes);
            current = network.output;
            break;
          }

          case 'easu': {
            const out = texture(stage.size, storage);
            const pipeline = await computePipeline(device, EASU_WGSL);
            passes.push({
              label: 'easu',
              pipeline,
              workgroups: workgroupsFor(stage.size),
              bindGroup: device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                  { binding: 0, resource: current.createView() },
                  { binding: 1, resource: out.createView() },
                ],
              }),
            });
            current = out;
            break;
          }

          case 'rcas': {
            const out = texture(stage.size, storage);
            const params = new Float32Array([rcasSharpness(tuning.sharpness), 0, 0, 0]);
            const pipeline = await computePipeline(device, RCAS_WGSL);
            passes.push({
              label: 'rcas',
              pipeline,
              workgroups: workgroupsFor(stage.size),
              bindGroup: device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                  { binding: 0, resource: current.createView() },
                  { binding: 1, resource: out.createView() },
                  { binding: 2, resource: { buffer: buffer(params, GPUBufferUsage.UNIFORM) } },
                ],
              }),
            });
            current = out;
            break;
          }
        }
      }
      return new FramePipeline(input, current, plan, passes, owned);
    } catch (error) {
      for (const resource of owned) resource.destroy();
      throw error;
    }
  }

  /** Records every pass into one compute pass. Each dispatch sees the previous one's writes. */
  encode(encoder: GPUCommandEncoder): void {
    if (this.passes.length === 0) return;
    const pass = encoder.beginComputePass();
    for (const { pipeline, bindGroup, workgroups } of this.passes) {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(...workgroups);
    }
    pass.end();
  }

  destroy(): void {
    for (const resource of this.owned.splice(0)) {
      try {
        resource.destroy();
      } catch {
        // Already gone with the device.
      }
    }
  }
}

interface Allocators {
  texture(size: Size, usage: number, layers?: number): GPUTexture;
  buffer(data: ArrayBufferView<ArrayBuffer>, usage: number): GPUBuffer;
}

/** Luma in, one pass per convolution, depth-to-space plus residual out. */
async function modelPasses(
  device: GPUDevice,
  model: ModelWeights,
  f16: boolean,
  source: GPUTexture,
  outSize: Size,
  { texture, buffer }: Allocators,
): Promise<{ passes: Pass[]; output: GPUTexture }> {
  const native = { width: source.width, height: source.height };
  const storage =
    GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC;
  const passes: Pass[] = [];

  const luma = texture(native, storage, 1);
  const lumaPipeline = await computePipeline(device, lumaWgsl(model.luma));
  passes.push({
    label: 'luma',
    pipeline: lumaPipeline,
    workgroups: workgroupsFor(native),
    bindGroup: device.createBindGroup({
      layout: lumaPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: source.createView() },
        { binding: 1, resource: luma.createView(arrayView) },
      ],
    }),
  });

  let activations = luma;
  for (const layer of convLayers(model)) {
    const inputGroups = groups(layer.in);
    const outputGroups = groups(layer.out);
    const out = texture(native, storage, outputGroups);
    const weights = packConvWeights(layer);
    const bias = packBias(layer);
    const pipeline = await computePipeline(
      device,
      convWgsl(inputGroups, outputGroups, layer.relu, f16),
    );
    passes.push({
      label: `conv ${layer.in}->${layer.out}`,
      pipeline,
      workgroups: workgroupsFor(native, outputGroups),
      bindGroup: device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: activations.createView(arrayView) },
          { binding: 1, resource: out.createView(arrayView) },
          {
            binding: 2,
            resource: {
              buffer: buffer(f16 ? toHalfArray(weights) : weights, GPUBufferUsage.STORAGE),
            },
          },
          {
            binding: 3,
            resource: { buffer: buffer(f16 ? toHalfArray(bias) : bias, GPUBufferUsage.STORAGE) },
          },
        ],
      }),
    });
    activations = out;
  }

  const out = texture(outSize, storage);
  const pipeline = await computePipeline(device, reconstructWgsl(model.scale));
  passes.push({
    label: 'reconstruct',
    pipeline,
    workgroups: workgroupsFor(outSize),
    bindGroup: device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: source.createView() },
        {
          binding: 1,
          resource: device.createSampler({ magFilter: 'linear', minFilter: 'linear' }),
        },
        { binding: 2, resource: activations.createView(arrayView) },
        { binding: 3, resource: out.createView() },
      ],
    }),
  });
  return { passes, output: out };
}
