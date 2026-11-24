/**
 * The graph weights format: a small DAG of named tensors, for networks that are more than a stack
 * of convolutions (skip connections, channel attention). `ultra` (EfRLFN) uses it. Written for this
 * project (MIT).
 *
 * ```json
 * { "arch": "efrlfn", "scale": 2, "input": "rgb", "output": "rgb",
 *   "nodes": [
 *     { "name": "conv_1", "op": "conv", "inputs": ["input"], "in": 3, "out": 52, "kernel": 3,
 *       "padding": "zero", "activation": "none", "weights": ..., "bias": ... },
 *     { "name": "b1.skip", "op": "add", "inputs": ["b1.c3_r", "conv_1"] },
 *     { "name": "b1.pool", "op": "global_avg_pool", "inputs": ["b1.c5"] },
 *     { "name": "b1.attention", "op": "channel_conv", "inputs": ["b1.pool"], "kernel": 3,
 *       "activation": "sigmoid", "weights": [...] },
 *     { "name": "b1", "op": "mul", "inputs": ["b1.c5", "b1.attention"] },
 *     ...
 *     { "name": "pixel_shuffle", "op": "depth_to_space", "inputs": ["upsampler"], "scale": 2 } ] }
 * ```
 *
 * Every node names its inputs; `input` is the frame. Nodes are listed in order, so each input is
 * the frame or an earlier node. Tensors are either **maps** (channels x height x width, at the
 * source resolution) or **vectors** (one value per channel, what a global pool produces).
 *
 * - `conv`: a same-size cross-correlation with a `kernel` x `kernel` window (1, 3, 5 or 7),
 *   `dilation` (default 1), `padding` `zero` (default, PyTorch's) or `replicate`, then
 *   `activation`. `weights` are `[out, in, k, k]` (PyTorch's OIHW, flattened), `bias` `[out]`.
 * - `add`, `mul`: elementwise; the second operand is a map of the same shape or a vector with as
 *   many channels (broadcast over the frame). An optional `activation` follows.
 * - `activation`: `none`, `relu`, `lrelu` (`slope`, default 0.01), `sigmoid` or `tanh`.
 * - `global_avg_pool`: map -> vector, the mean of each channel over the whole frame.
 * - `channel_conv`: vector -> vector, a 1-D convolution across the channels (ECA's `Conv1d(1, 1,
 *   k)`), zero padded, with an optional single `bias`, then `activation`.
 * - `depth_to_space`: `nn.PixelShuffle(scale)`. It is the last node and only that.
 *
 * `input` is `rgb` (R'G'B' in 0..1, 3 channels) or `luma` (Y' with the `luma` coefficients,
 * 1 channel). `output` is `rgb` (the last node gives the picture, 3 channels) or `luma_residual`
 * (1 channel added to a bilinear upscale, as in the linear format).
 *
 * A tensor's values are a list of numbers or `{ "dtype": "float16" | "float32", "base64": ... }`
 * (little endian), see `decodeTensor`.
 */

import {
  decodeTensor,
  groups,
  isChannelCount,
  LUMA_WEIGHTS,
  MAX_CHANNELS,
  ModelError,
  type LumaCoefficients,
} from './tensor.js';

export type Activation = 'none' | 'relu' | 'lrelu' | 'sigmoid' | 'tanh';
export type Padding = 'zero' | 'replicate';

interface NodeBase {
  name: string;
  inputs: string[];
}

export interface ConvNode extends NodeBase {
  op: 'conv';
  in: number;
  out: number;
  kernel: number;
  dilation: number;
  padding: Padding;
  activation: Activation;
  slope: number;
  /** `[out, in, kernel, kernel]`. */
  weights: Float32Array;
  /** `[out]`. */
  bias: Float32Array;
}

export interface BinaryNode extends NodeBase {
  op: 'add' | 'mul';
  activation: Activation;
  slope: number;
}

export interface ActivationNode extends NodeBase {
  op: 'activation';
  activation: Activation;
  slope: number;
}

export interface GlobalAvgPoolNode extends NodeBase {
  op: 'global_avg_pool';
}

export interface ChannelConvNode extends NodeBase {
  op: 'channel_conv';
  kernel: number;
  weights: Float32Array;
  bias: number;
  activation: Activation;
  slope: number;
}

export interface GraphDepthToSpaceNode extends NodeBase {
  op: 'depth_to_space';
  scale: number;
}

export type GraphNode =
  | ConvNode
  | BinaryNode
  | ActivationNode
  | GlobalAvgPoolNode
  | ChannelConvNode
  | GraphDepthToSpaceNode;

export type GraphInput = 'rgb' | 'luma';
export type GraphOutput = 'rgb' | 'luma_residual';

/** A network in the graph format, validated. */
export interface GraphModel {
  arch: string;
  scale: number;
  /** Widest tensor; informational. */
  channels: number;
  input: GraphInput;
  output: GraphOutput;
  /** Coefficients for a `luma` input or a `luma_residual` output. Default `bt709`. */
  luma: LumaCoefficients;
  nodes: GraphNode[];
}

/** A tensor's shape: channels, and whether it is a per-channel vector instead of a map. */
export interface TensorShape {
  channels: number;
  vector: boolean;
}

export const GRAPH_INPUT = 'input';
export const MAX_NODES = 256;
export const GRAPH_SCALES: readonly number[] = [2, 3, 4];
const KERNELS: readonly number[] = [1, 3, 5, 7];
const ACTIVATIONS: readonly Activation[] = ['none', 'relu', 'lrelu', 'sigmoid', 'tanh'];
const OPS: readonly string[] = [
  'conv',
  'add',
  'mul',
  'activation',
  'global_avg_pool',
  'channel_conv',
  'depth_to_space',
];

export function isGraphModel(model: object): model is GraphModel {
  return Array.isArray((model as { nodes?: unknown }).nodes);
}

function integer(value: unknown, min: number, max: number): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= min && value <= max;
}

function activationOf(node: Record<string, unknown>, where: string) {
  const activation = node.activation === undefined ? 'none' : node.activation;
  if (!ACTIVATIONS.includes(activation as Activation)) {
    throw new ModelError(`${where}: activation must be one of ${ACTIVATIONS.join(', ')}`);
  }
  const slope = node.slope === undefined ? 0.01 : node.slope;
  if (typeof slope !== 'number' || !Number.isFinite(slope)) {
    throw new ModelError(`${where}: slope must be a finite number`);
  }
  return { activation: activation as Activation, slope };
}

/** Checks a graph weights file and returns it typed. Throws `ModelError` naming the first problem. */
export function parseGraphModel(json: unknown): GraphModel {
  if (!json || typeof json !== 'object') throw new ModelError('weights must be a JSON object');
  const o = json as Record<string, unknown>;
  const scale = o.scale;
  if (typeof scale !== 'number' || !GRAPH_SCALES.includes(scale)) {
    throw new ModelError(`scale must be one of ${GRAPH_SCALES.join(', ')}`);
  }
  const input = o.input === undefined ? 'rgb' : o.input;
  if (input !== 'rgb' && input !== 'luma') throw new ModelError('input must be rgb or luma');
  const output = o.output === undefined ? 'rgb' : o.output;
  if (output !== 'rgb' && output !== 'luma_residual') {
    throw new ModelError('output must be rgb or luma_residual');
  }
  const luma = o.luma === undefined ? 'bt709' : o.luma;
  if (luma !== 'bt709' && luma !== 'bt601') throw new ModelError('luma must be bt709 or bt601');
  const nodes = o.nodes;
  if (!Array.isArray(nodes) || nodes.length < 2) {
    throw new ModelError('nodes must list at least one conv and a depth_to_space');
  }
  if (nodes.length > MAX_NODES) throw new ModelError(`at most ${MAX_NODES} nodes`);

  const shapes = new Map<string, TensorShape>([
    [GRAPH_INPUT, { channels: input === 'rgb' ? 3 : 1, vector: false }],
  ]);
  const parsed: GraphNode[] = [];
  let widest = 0;

  nodes.forEach((raw: unknown, index) => {
    if (!raw || typeof raw !== 'object') throw new ModelError(`nodes[${index}] is not an object`);
    const node = raw as Record<string, unknown>;
    const { name, op } = node;
    if (typeof name !== 'string' || name === '') {
      throw new ModelError(`nodes[${index}]: name must be a non-empty string`);
    }
    const where = `nodes[${index}] (${name})`;
    if (shapes.has(name)) throw new ModelError(`${where}: the name is already taken`);
    if (typeof op !== 'string' || !OPS.includes(op)) {
      throw new ModelError(`${where}: unknown op ${String(op)}`);
    }
    const arity = op === 'add' || op === 'mul' ? 2 : 1;
    const inputs = node.inputs;
    if (
      !Array.isArray(inputs) ||
      inputs.length !== arity ||
      !inputs.every((n) => typeof n === 'string')
    ) {
      throw new ModelError(`${where}: inputs must name ${arity} tensor${arity > 1 ? 's' : ''}`);
    }
    const ins = inputs.map((n: string) => {
      const shape = shapes.get(n);
      if (!shape) throw new ModelError(`${where}: input ${n} is not the frame or an earlier node`);
      return shape;
    });
    const map = (i: number) => {
      if (ins[i].vector) throw new ModelError(`${where}: input ${inputs[i]} must be a map`);
      return ins[i];
    };
    const last = index === nodes.length - 1;
    if (last !== (op === 'depth_to_space')) {
      throw new ModelError(`${where}: depth_to_space must be the last node, and only that`);
    }

    let shape: TensorShape;
    switch (op) {
      case 'conv': {
        const { in: cin, out: cout, kernel } = node;
        if (!isChannelCount(cin) || !isChannelCount(cout)) {
          throw new ModelError(`${where}: in and out must be integers in 1..${MAX_CHANNELS}`);
        }
        if (cin !== map(0).channels) {
          throw new ModelError(`${where}: in is ${cin}, but ${inputs[0]} has ${ins[0].channels}`);
        }
        if (typeof kernel !== 'number' || !KERNELS.includes(kernel)) {
          throw new ModelError(`${where}: kernel must be one of ${KERNELS.join(', ')}`);
        }
        const dilation = node.dilation === undefined ? 1 : node.dilation;
        if (!integer(dilation, 1, 8)) throw new ModelError(`${where}: dilation must be 1..8`);
        const padding = node.padding === undefined ? 'zero' : node.padding;
        if (padding !== 'zero' && padding !== 'replicate') {
          throw new ModelError(`${where}: padding must be zero or replicate`);
        }
        const weights = decodeTensor(
          node.weights,
          cout * cin * kernel * kernel,
          `${where}: weights`,
        );
        const bias =
          node.bias === undefined
            ? new Float32Array(cout)
            : decodeTensor(node.bias, cout, `${where}: bias`);
        parsed.push({
          name,
          op,
          inputs,
          in: cin,
          out: cout,
          kernel,
          dilation,
          padding,
          ...activationOf(node, where),
          weights,
          bias,
        });
        shape = { channels: cout, vector: false };
        break;
      }
      case 'add':
      case 'mul': {
        const a = map(0);
        if (ins[1].channels !== a.channels) {
          throw new ModelError(`${where}: ${inputs[0]} and ${inputs[1]} differ in channels`);
        }
        parsed.push({ name, op, inputs, ...activationOf(node, where) });
        shape = { channels: a.channels, vector: false };
        break;
      }
      case 'activation': {
        const a = map(0);
        parsed.push({ name, op, inputs, ...activationOf(node, where) });
        shape = { channels: a.channels, vector: false };
        break;
      }
      case 'global_avg_pool': {
        parsed.push({ name, op, inputs });
        shape = { channels: map(0).channels, vector: true };
        break;
      }
      case 'channel_conv': {
        if (!ins[0].vector) throw new ModelError(`${where}: input ${inputs[0]} must be a vector`);
        const kernel = node.kernel;
        if (!integer(kernel, 1, 15) || kernel % 2 === 0) {
          throw new ModelError(`${where}: kernel must be odd, 1..15`);
        }
        const weights = decodeTensor(node.weights, kernel, `${where}: weights`);
        const bias = node.bias === undefined ? 0 : decodeTensor(node.bias, 1, `${where}: bias`)[0];
        parsed.push({ name, op, inputs, kernel, weights, bias, ...activationOf(node, where) });
        shape = { ...ins[0] };
        break;
      }
      case 'depth_to_space': {
        if (node.scale !== scale) throw new ModelError(`${where}: scale must match the model's`);
        const planes = output === 'rgb' ? 3 : 1;
        if (map(0).channels !== planes * scale * scale) {
          throw new ModelError(
            `${where}: an ${output} output needs ${planes * scale * scale} channels, got ${ins[0].channels}`,
          );
        }
        parsed.push({ name, op, inputs, scale });
        shape = { channels: planes, vector: false };
        break;
      }
      default:
        throw new ModelError(`${where}: unknown op ${String(op)}`);
    }
    shapes.set(name, shape);
    widest = Math.max(widest, shape.channels);
  });

  return {
    arch: typeof o.arch === 'string' ? o.arch : 'unknown',
    scale,
    channels: isChannelCount(o.channels) ? o.channels : widest,
    input,
    output,
    luma,
    nodes: parsed,
  };
}

/** Shape of the frame and of every node's output, by name. */
export function tensorShapes(model: GraphModel): Map<string, TensorShape> {
  const shapes = new Map<string, TensorShape>([
    [GRAPH_INPUT, { channels: model.input === 'rgb' ? 3 : 1, vector: false }],
  ]);
  for (const node of model.nodes) {
    const first = shapes.get(node.inputs[0]) as TensorShape;
    switch (node.op) {
      case 'conv':
        shapes.set(node.name, { channels: node.out, vector: false });
        break;
      case 'global_avg_pool':
        shapes.set(node.name, { channels: first.channels, vector: true });
        break;
      case 'depth_to_space':
        shapes.set(node.name, { channels: model.output === 'rgb' ? 3 : 1, vector: false });
        break;
      default:
        shapes.set(node.name, { ...first });
    }
  }
  return shapes;
}

/** Multiply-accumulates per source pixel, the cost that dominates a frame. */
export function macsPerPixel(model: GraphModel): number {
  let macs = 0;
  for (const node of model.nodes) {
    if (node.op === 'conv') macs += node.in * node.out * node.kernel * node.kernel;
  }
  return macs;
}

/** Number of weights and biases in the file. */
export function parameterCount(model: GraphModel): number {
  let count = 0;
  for (const node of model.nodes) {
    if (node.op === 'conv') count += node.weights.length + node.bias.length;
    if (node.op === 'channel_conv') count += node.weights.length;
  }
  return count;
}

/**
 * Packs a convolution's weights for the WGSL shader: one `mat4x4` per output group, input group and
 * tap, laid out `[og][ig][tap]`, taps row by row (`tap = ky * kernel + kx`). Column `k` of a matrix
 * holds the weights from input channel `ig*4 + k` to output channels `og*4 .. og*4 + 3`. Output
 * groups are padded with zeros up to `outputGroups` (a multiple of what one invocation computes).
 */
export function packConvKernel(
  node: ConvNode,
  outputGroups = groups(node.out),
): Float32Array<ArrayBuffer> {
  const ig = groups(node.in);
  const taps = node.kernel * node.kernel;
  const packed = new Float32Array(outputGroups * ig * taps * 16);
  for (let o = 0; o < node.out; o++) {
    for (let i = 0; i < node.in; i++) {
      for (let t = 0; t < taps; t++) {
        const matrix = ((o >> 2) * ig + (i >> 2)) * taps + t;
        packed[matrix * 16 + (i & 3) * 4 + (o & 3)] = node.weights[(o * node.in + i) * taps + t];
      }
    }
  }
  return packed;
}

/** A convolution's bias as one `vec4` per output group, zero-padded to `outputGroups`. */
export function packConvBias(
  node: ConvNode,
  outputGroups = groups(node.out),
): Float32Array<ArrayBuffer> {
  const packed = new Float32Array(outputGroups * 4);
  packed.set(node.bias);
  return packed;
}

// ---- CPU reference ------------------------------------------------------------------------------

/** One plane per channel, `width * height` values each. */
type Planes = Float32Array[];

/** A map (planes) or a vector (one value per channel). */
export type GraphTensor = { planes: Planes } | { vector: Float32Array };

export function activate(value: number, activation: Activation, slope: number): number {
  switch (activation) {
    case 'relu':
      return Math.max(0, value);
    case 'lrelu':
      return value >= 0 ? value : value * slope;
    case 'sigmoid':
      return 1 / (1 + Math.exp(-value));
    case 'tanh':
      return Math.tanh(value);
    default:
      return value;
  }
}

/** `conv` straight from the PyTorch layout. */
export function graphConvReference(
  node: ConvNode,
  input: Planes,
  width: number,
  height: number,
): Planes {
  const { kernel: k, dilation: d } = node;
  const r = (d * (k - 1)) / 2;
  const output: Planes = [];
  for (let o = 0; o < node.out; o++) {
    const plane = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sum = node.bias[o];
        for (let ky = 0; ky < k; ky++) {
          let sy = y + ky * d - r;
          if (node.padding === 'replicate') sy = Math.min(height - 1, Math.max(0, sy));
          else if (sy < 0 || sy >= height) continue;
          for (let kx = 0; kx < k; kx++) {
            let sx = x + kx * d - r;
            if (node.padding === 'replicate') sx = Math.min(width - 1, Math.max(0, sx));
            else if (sx < 0 || sx >= width) continue;
            for (let i = 0; i < node.in; i++) {
              sum +=
                node.weights[((o * node.in + i) * k + ky) * k + kx] * input[i][sy * width + sx];
            }
          }
        }
        plane[y * width + x] = activate(sum, node.activation, node.slope);
      }
    }
    output.push(plane);
  }
  return output;
}

/**
 * The same convolution computed the way the WGSL shader does it, from `packConvKernel` matrices
 * and channel groups. Proves the packing.
 */
export function graphConvPackedReference(
  node: ConvNode,
  packed: Float32Array,
  bias: Float32Array,
  input: Planes,
  width: number,
  height: number,
): Planes {
  const ig = groups(node.in);
  const og = bias.length / 4;
  const { kernel: k, dilation: d } = node;
  const r = (d * (k - 1)) / 2;
  const texel = (channel: number, x: number, y: number) =>
    channel < input.length ? input[channel][y * width + x] : 0;
  const output: Planes = Array.from({ length: og * 4 }, () => new Float32Array(width * height));
  for (let g = 0; g < og; g++) {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const acc = Array.from(bias.subarray(g * 4, g * 4 + 4));
        for (let t = 0; t < k * k; t++) {
          let qx = x + (t % k) * d - r;
          let qy = y + Math.floor(t / k) * d - r;
          if (node.padding === 'replicate') {
            qx = Math.min(width - 1, Math.max(0, qx));
            qy = Math.min(height - 1, Math.max(0, qy));
          } else if (qx < 0 || qy < 0 || qx >= width || qy >= height) {
            continue;
          }
          for (let i = 0; i < ig; i++) {
            const m = ((g * ig + i) * k * k + t) * 16;
            for (let c = 0; c < 4; c++) {
              const a = texel(i * 4 + c, qx, qy);
              for (let j = 0; j < 4; j++) acc[j] += packed[m + c * 4 + j] * a;
            }
          }
        }
        for (let j = 0; j < 4; j++) {
          output[g * 4 + j][y * width + x] = activate(acc[j], node.activation, node.slope);
        }
      }
    }
  }
  return output;
}

/** The frame as the network's input planes: R, G, B, or the luma. */
export function graphInputReference(
  model: GraphModel,
  rgb: readonly [Float32Array, Float32Array, Float32Array],
): Planes {
  if (model.input === 'rgb') return [...rgb];
  const [r, g, b] = LUMA_WEIGHTS[model.luma];
  return [rgb[0].map((v, i) => r * v + g * rgb[1][i] + b * rgb[2][i])];
}

/**
 * Runs a graph on the CPU. `input` are the network's input planes (see `graphInputReference`);
 * the result is the last node's planes at `scale` times the size, unclamped: the picture for an
 * `rgb` output, the residual for `luma_residual`.
 */
export function runGraphReference(
  model: GraphModel,
  input: Planes,
  width: number,
  height: number,
): Planes {
  const tensors = new Map<string, GraphTensor>([[GRAPH_INPUT, { planes: input }]]);
  const planesOf = (name: string) => (tensors.get(name) as { planes: Planes }).planes;
  let result: Planes = [];
  for (const node of model.nodes) {
    let out: GraphTensor;
    switch (node.op) {
      case 'conv':
        out = { planes: graphConvReference(node, planesOf(node.inputs[0]), width, height) };
        break;
      case 'add':
      case 'mul': {
        const a = planesOf(node.inputs[0]);
        const b = tensors.get(node.inputs[1]) as GraphTensor;
        out = {
          planes: a.map((plane, c) =>
            plane.map((v, i) => {
              const w = 'vector' in b ? b.vector[c] : b.planes[c][i];
              return activate(node.op === 'add' ? v + w : v * w, node.activation, node.slope);
            }),
          ),
        };
        break;
      }
      case 'activation':
        out = {
          planes: planesOf(node.inputs[0]).map((plane) =>
            plane.map((v) => activate(v, node.activation, node.slope)),
          ),
        };
        break;
      case 'global_avg_pool':
        out = {
          vector: Float32Array.from(planesOf(node.inputs[0]), (plane) => {
            let sum = 0;
            for (const v of plane) sum += v;
            return sum / plane.length;
          }),
        };
        break;
      case 'channel_conv': {
        const v = (tensors.get(node.inputs[0]) as { vector: Float32Array }).vector;
        const half = (node.kernel - 1) / 2;
        out = {
          vector: v.map((_, c) => {
            let sum = node.bias;
            for (let j = 0; j < node.kernel; j++) {
              const src = c + j - half;
              if (src >= 0 && src < v.length) sum += node.weights[j] * v[src];
            }
            return activate(sum, node.activation, node.slope);
          }),
        };
        break;
      }
      case 'depth_to_space': {
        const src = planesOf(node.inputs[0]);
        const s = node.scale;
        const outWidth = width * s;
        result = Array.from({ length: src.length / (s * s) }, (_, c) => {
          const plane = new Float32Array(outWidth * height * s);
          for (let y = 0; y < height * s; y++) {
            for (let x = 0; x < outWidth; x++) {
              const channel = c * s * s + (y % s) * s + (x % s);
              plane[y * outWidth + x] = src[channel][Math.floor(y / s) * width + Math.floor(x / s)];
            }
          }
          return plane;
        });
        out = { planes: result };
        break;
      }
    }
    tensors.set(node.name, out);
  }
  return result;
}
