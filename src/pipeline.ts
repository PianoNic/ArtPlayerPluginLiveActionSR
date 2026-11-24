import type { Size } from './layout.js';
import {
  GRAPH_INPUT,
  packConvBias,
  packConvKernel,
  tensorShapes,
  type ConvNode,
  type GraphModel,
} from './graph.js';
import { groups, isGraphModel, linearToGraph, toHalfArray, type ModelWeights } from './model.js';
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
import {
  GRAPH_POOL_FINAL_WGSL,
  GRAPH_POOL_PARTIAL_WGSL,
  graphChannelConvWgsl,
  graphConvWgsl,
  graphElementwiseWgsl,
  graphInputWgsl,
  graphRgbOutputWgsl,
  type ActivationFormat,
  POOL_TILE,
  reconstructWgsl,
} from './shaders/graph.wgsl.js';

export interface PipelineConfig {
  preset: Preset;
  native: Size;
  target: Size;
  tuning: Tuning;
  /** Weights for `balanced` / `quality` / `ultra`; null runs the `performance` stages. */
  model: ModelWeights | null;
  /** Run the CNN in half precision (`shader-f16`). */
  f16: boolean;
  /** How graph convolutions are split up; the defaults are `graphKernelDefaults()`. */
  graphKernel?: Partial<GraphKernelOptions>;
}

export interface GraphKernelOptions {
  /** Output channel groups one convolution invocation computes (more: fewer input loads). */
  perInvocation: number;
  /** Horizontally adjacent pixels one convolution invocation computes (more: fewer weight loads). */
  pixels: number;
  /** On the `f16` path, accumulate the products in `f32` (slower, a little more exact). */
  accumulateF32: boolean;
  /** Storage of the activation maps between passes. */
  format: ActivationFormat;
}

/**
 * Measured on an RTX 5080 with EfRLFN at 1280x720 (see demo/model-check.html): 8 pixels x 2 output
 * groups per invocation runs 2.7x faster than one pixel per invocation (16 pixels gains another 6%,
 * at twice the registers). Accumulating in f16 would save 15% but triples the worst-case error
 * (1.2/255 instead of 0.47/255 against PyTorch), so the f16 path multiplies in f16 and adds in f32.
 */
export function graphKernelDefaults(): GraphKernelOptions {
  return { perInvocation: 2, pixels: 8, accumulateF32: true, format: 'rgba16float' };
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

    const texture = (
      size: Size,
      usage: number,
      layers = 1,
      format: GPUTextureFormat = FORMAT,
    ): GPUTexture => {
      const created = device.createTexture({
        size: [size.width, size.height, layers],
        format,
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
            // A linear model runs as the equivalent graph: same convolutions, faster kernel.
            const graph = isGraphModel(model) ? model : linearToGraph(model);
            const network = await graphPasses(device, graph, config, current, stage.size, {
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

  /**
   * Records every pass into one compute pass. Each dispatch sees the previous one's writes.
   * `timestampWrites` (needs `timestamp-query`) times that compute pass; the benchmark page uses it.
   */
  encode(encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites): void {
    if (this.passes.length === 0) return;
    const pass = encoder.beginComputePass(timestampWrites ? { timestampWrites } : undefined);
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
  texture(size: Size, usage: number, layers?: number, format?: GPUTextureFormat): GPUTexture;
  buffer(data: ArrayBufferView<ArrayBuffer>, usage: number): GPUBuffer;
}

/**
 * A graph network: the frame into a texture array, one pass per node (two for a global pool), and
 * the output. Map textures are recycled as soon as their last reader has run, so a deep network
 * needs only a few frames' worth of activations at a time.
 */
async function graphPasses(
  device: GPUDevice,
  model: GraphModel,
  config: PipelineConfig,
  source: GPUTexture,
  outSize: Size,
  { texture, buffer }: Allocators,
): Promise<{ passes: Pass[]; output: GPUTexture }> {
  const kernel = { ...graphKernelDefaults(), ...config.graphKernel };
  const native = { width: source.width, height: source.height };
  const storage =
    GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC;
  const shapes = tensorShapes(model);
  const passes: Pass[] = [];

  const lastUse = new Map<string, number>();
  model.nodes.forEach((node, index) => {
    for (const name of node.inputs) lastUse.set(name, index);
  });

  // Free map textures by layer count. A node's output is taken before its inputs are returned,
  // so no pass reads and writes the same texture.
  const pool = new Map<number, GPUTexture[]>();
  const take = (layers: number): GPUTexture =>
    pool.get(layers)?.pop() ?? texture(native, storage, layers, kernel.format);
  const give = (tex: GPUTexture) => {
    const free = pool.get(tex.depthOrArrayLayers) ?? [];
    free.push(tex);
    pool.set(tex.depthOrArrayLayers, free);
  };
  const maps = new Map<string, GPUTexture>();
  const vectors = new Map<string, GPUBuffer>();
  const vectorBuffer = (channels: number) =>
    buffer(new Float32Array(groups(channels) * 4), GPUBufferUsage.STORAGE);
  const bind = (pipeline: GPUComputePipeline, resources: GPUBindingResource[]) =>
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: resources.map((resource, binding) => ({ binding, resource })),
    });

  // The frame, as the network's input channels.
  {
    const out = take(1);
    const pipeline = await computePipeline(
      device,
      graphInputWgsl(model.input, model.luma, kernel.format),
    );
    passes.push({
      label: model.input,
      pipeline,
      workgroups: workgroupsFor(native),
      bindGroup: bind(pipeline, [source.createView(), out.createView(arrayView)]),
    });
    maps.set(GRAPH_INPUT, out);
  }

  let last: GPUTexture | null = null;
  for (const [index, node] of model.nodes.entries()) {
    const shape = shapes.get(node.name)!;
    const inputMap = maps.get(node.inputs[0]);
    switch (node.op) {
      case 'conv': {
        const per = Math.min(kernel.perInvocation, groups(node.out));
        const chunks = Math.ceil(groups(node.out) / per);
        const outputGroups = chunks * per;
        const out = take(groups(node.out));
        const weights = packConvKernel(node, outputGroups);
        const bias = packConvBias(node, outputGroups);
        const pipeline = await computePipeline(
          device,
          graphConvWgsl({
            inputGroups: groups(node.in),
            outputGroups,
            storedGroups: groups(node.out),
            perInvocation: per,
            pixels: kernel.pixels,
            kernel: node.kernel,
            dilation: node.dilation,
            padding: node.padding,
            activation: node.activation,
            slope: node.slope,
            f16: config.f16,
            accumulateF32: kernel.accumulateF32,
            format: kernel.format,
          }),
        );
        passes.push({
          label: convLabel(node),
          pipeline,
          workgroups: [
            Math.ceil(native.width / (WORKGROUP * kernel.pixels)),
            Math.ceil(native.height / WORKGROUP),
            chunks,
          ],
          bindGroup: bind(pipeline, [
            inputMap!.createView(arrayView),
            out.createView(arrayView),
            { buffer: buffer(config.f16 ? toHalfArray(weights) : weights, GPUBufferUsage.STORAGE) },
            { buffer: buffer(config.f16 ? toHalfArray(bias) : bias, GPUBufferUsage.STORAGE) },
          ]),
        });
        maps.set(node.name, out);
        break;
      }
      case 'add':
      case 'mul':
      case 'activation': {
        const out = take(groups(shape.channels));
        const second = node.op === 'activation' ? undefined : node.inputs[1];
        const vector = second !== undefined && vectors.has(second);
        const pipeline = await computePipeline(
          device,
          graphElementwiseWgsl(
            node.op === 'activation' ? null : node.op,
            vector,
            node.activation,
            node.slope,
            kernel.format,
          ),
        );
        const resources: GPUBindingResource[] = [
          inputMap!.createView(arrayView),
          out.createView(arrayView),
        ];
        if (second !== undefined) {
          resources.push(
            vector ? { buffer: vectors.get(second)! } : maps.get(second)!.createView(arrayView),
          );
        }
        passes.push({
          label: node.op,
          pipeline,
          workgroups: workgroupsFor(native, groups(shape.channels)),
          bindGroup: bind(pipeline, resources),
        });
        maps.set(node.name, out);
        break;
      }
      case 'global_avg_pool': {
        const layers = groups(shape.channels);
        const tilesX = Math.ceil(native.width / POOL_TILE);
        const tilesY = Math.ceil(native.height / POOL_TILE);
        const partial = buffer(
          new Float32Array(layers * tilesX * tilesY * 4),
          GPUBufferUsage.STORAGE,
        );
        const result = vectorBuffer(shape.channels);
        const params = new ArrayBuffer(16);
        new Uint32Array(params, 0, 1)[0] = tilesX * tilesY;
        new Float32Array(params, 4, 1)[0] = 1 / (native.width * native.height);
        const first = await computePipeline(device, GRAPH_POOL_PARTIAL_WGSL);
        const second = await computePipeline(device, GRAPH_POOL_FINAL_WGSL);
        passes.push(
          {
            label: 'pool',
            pipeline: first,
            workgroups: [tilesX, tilesY, layers],
            bindGroup: bind(first, [inputMap!.createView(arrayView), { buffer: partial }]),
          },
          {
            label: 'pool sum',
            pipeline: second,
            workgroups: [layers, 1, 1],
            bindGroup: bind(second, [
              { buffer: partial },
              { buffer: result },
              { buffer: buffer(new Uint8Array(params), GPUBufferUsage.UNIFORM) },
            ]),
          },
        );
        vectors.set(node.name, result);
        break;
      }
      case 'channel_conv': {
        const result = vectorBuffer(shape.channels);
        const weights = new Float32Array(node.kernel + 1);
        weights.set(node.weights);
        weights[node.kernel] = node.bias;
        const pipeline = await computePipeline(
          device,
          graphChannelConvWgsl(shape.channels, node.kernel, node.activation, node.slope),
        );
        passes.push({
          label: 'channel conv',
          pipeline,
          workgroups: [Math.ceil(shape.channels / 64), 1, 1],
          bindGroup: bind(pipeline, [
            { buffer: vectors.get(node.inputs[0])! },
            { buffer: result },
            { buffer: buffer(weights, GPUBufferUsage.STORAGE) },
          ]),
        });
        vectors.set(node.name, result);
        break;
      }
      case 'depth_to_space':
        // The output pass below reads the last convolution directly.
        last = inputMap!;
        break;
    }
    if (node.op === 'depth_to_space') continue;
    for (const name of new Set(node.inputs)) {
      const map = maps.get(name);
      if (map && lastUse.get(name) === index) {
        give(map);
        maps.delete(name);
      }
    }
  }

  const out = texture(outSize, storage);
  if (model.output === 'rgb') {
    const pipeline = await computePipeline(device, graphRgbOutputWgsl(model.scale));
    passes.push({
      label: 'rgb out',
      pipeline,
      workgroups: workgroupsFor(outSize),
      bindGroup: bind(pipeline, [last!.createView(arrayView), out.createView()]),
    });
  } else {
    const pipeline = await computePipeline(device, reconstructWgsl(model.scale));
    passes.push({
      label: 'reconstruct',
      pipeline,
      workgroups: workgroupsFor(outSize),
      bindGroup: bind(pipeline, [
        source.createView(),
        device.createSampler({ magFilter: 'linear', minFilter: 'linear' }),
        last!.createView(arrayView),
        out.createView(),
      ]),
    });
  }
  return { passes, output: out };
}

function convLabel(node: ConvNode): string {
  return `${node.name} ${node.in}->${node.out} k${node.kernel}`;
}
