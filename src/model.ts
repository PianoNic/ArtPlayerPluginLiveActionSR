/**
 * The networks behind the CNN presets. Two weight formats are accepted:
 *
 * - the **linear** format below, a plain stack of 3x3 convolutions on luma (`balanced`,
 *   `quality`);
 * - the **graph** format of `graph.ts`, a small DAG of named tensors for networks with skips and
 *   attention (`ultra`, EfRLFN). A file with a `nodes` array is a graph.
 *
 * This module holds the linear format, its validation, its CPU reference and its conversion to a
 * graph (`linearToGraph`: the GPU runs both formats through the graph executor), plus
 * `parseModel` / `loadModel` for both.
 *
 * The network sees one channel, the luma (Y') of the source frame, and predicts a residual at
 * `scale` times the resolution. That residual is added to a bilinear upscale of the frame
 * (`F.interpolate(..., mode='bilinear', align_corners=False)` in PyTorch terms). Adding the same
 * value to R, G and B changes Y' by exactly that value and leaves the chroma alone.
 *
 * ```json
 * { "arch": "artcnn-like", "scale": 2, "channels": 16, "luma": "bt709",
 *   "layers": [
 *     { "type": "conv3x3", "in": 1,  "out": 16, "relu": true,  "weights": [...], "bias": [...] },
 *     { "type": "conv3x3", "in": 16, "out": 4,  "relu": false, "weights": [...], "bias": [...] },
 *     { "type": "depth_to_space", "scale": 2 } ] }
 * ```
 *
 * `weights` is PyTorch's `Conv2d.weight` flattened row-major, shape `[out, in, 3, 3]`, and `bias`
 * is `[out]`. Convolutions pad by clamping to the edge (replicate, as trained; see training/README.md). `depth_to_space` is
 * `nn.PixelShuffle(scale)`: output pixel `(x*s + dx, y*s + dy)` takes channel `dy*s + dx`. Values
 * are in 0..1 (8-bit levels divided by 255).
 */

import { GRAPH_INPUT, isGraphModel, parseGraphModel, type GraphModel } from './graph.js';
import {
  isChannelCount,
  isFiniteArray,
  MAX_CHANNELS,
  ModelError,
  type LumaCoefficients,
} from './tensor.js';

export {
  fromHalfBits,
  groups,
  LUMA_WEIGHTS,
  MAX_CHANNELS,
  ModelError,
  toHalfArray,
  toHalfBits,
  type LumaCoefficients,
} from './tensor.js';
export { isGraphModel };

export interface Conv3x3Layer {
  type: 'conv3x3';
  in: number;
  out: number;
  relu: boolean;
  weights: number[];
  bias: number[];
}

export interface DepthToSpaceLayer {
  type: 'depth_to_space';
  scale: number;
}

export type ModelLayer = Conv3x3Layer | DepthToSpaceLayer;

/** A network in the linear format. */
export interface LinearModel {
  arch: string;
  scale: number;
  /** Width of the hidden layers; informational. */
  channels: number;
  /** Default `bt709`. */
  luma: LumaCoefficients;
  layers: ModelLayer[];
}

/** Hard limit, so a broken file cannot generate a shader that takes minutes to compile. */
export const MAX_LAYERS = 32;
export const SUPPORTED_SCALES: readonly number[] = [2, 3, 4];

/** Weights in either format, validated. Tell them apart with `isGraphModel`. */
export type ModelWeights = LinearModel | GraphModel;

/** Checks a parsed weights file and returns it typed. Throws `ModelError` naming the first problem. */
export function parseModel(json: unknown): ModelWeights {
  if (!json || typeof json !== 'object') throw new ModelError('weights must be a JSON object');
  if (Array.isArray((json as { nodes?: unknown }).nodes)) return parseGraphModel(json);
  return parseLinearModel(json);
}

/** `parseModel` for the linear format only. */
export function parseLinearModel(json: unknown): LinearModel {
  if (!json || typeof json !== 'object') throw new ModelError('weights must be a JSON object');
  const o = json as Record<string, unknown>;
  const scale = o.scale;
  if (typeof scale !== 'number' || !SUPPORTED_SCALES.includes(scale)) {
    throw new ModelError(`scale must be one of ${SUPPORTED_SCALES.join(', ')}`);
  }
  const luma = o.luma === undefined ? 'bt709' : o.luma;
  if (luma !== 'bt709' && luma !== 'bt601') throw new ModelError('luma must be bt709 or bt601');
  const layers = o.layers;
  if (!Array.isArray(layers) || layers.length < 2) {
    throw new ModelError('layers must list at least one conv3x3 and a depth_to_space');
  }
  if (layers.length > MAX_LAYERS) throw new ModelError(`at most ${MAX_LAYERS} layers`);

  const parsed: ModelLayer[] = [];
  let channels = 1;
  let widest = 0;
  layers.forEach((raw: unknown, index) => {
    const where = `layers[${index}]`;
    if (!raw || typeof raw !== 'object') throw new ModelError(`${where} is not an object`);
    const layer = raw as Record<string, unknown>;
    const last = index === layers.length - 1;
    if (last) {
      if (layer.type !== 'depth_to_space') {
        throw new ModelError(`${where}: the last layer must be depth_to_space`);
      }
      if (layer.scale !== scale) throw new ModelError(`${where}: scale must match the model's`);
      if (channels !== scale * scale) {
        throw new ModelError(`${where}: needs ${scale * scale} input channels, got ${channels}`);
      }
      parsed.push({ type: 'depth_to_space', scale });
      return;
    }
    if (layer.type !== 'conv3x3') throw new ModelError(`${where}: expected conv3x3`);
    const { in: inputs, out: outputs } = layer;
    if (!isChannelCount(inputs) || !isChannelCount(outputs)) {
      throw new ModelError(`${where}: in and out must be integers in 1..${MAX_CHANNELS}`);
    }
    if (inputs !== channels) {
      throw new ModelError(`${where}: in is ${inputs}, but the previous layer has ${channels}`);
    }
    if (!isFiniteArray(layer.weights, outputs * inputs * 9)) {
      throw new ModelError(`${where}: weights must be ${outputs * inputs * 9} finite numbers`);
    }
    if (!isFiniteArray(layer.bias, outputs)) {
      throw new ModelError(`${where}: bias must be ${outputs} finite numbers`);
    }
    if (layer.relu !== undefined && typeof layer.relu !== 'boolean') {
      throw new ModelError(`${where}: relu must be a boolean`);
    }
    parsed.push({
      type: 'conv3x3',
      in: inputs,
      out: outputs,
      relu: layer.relu === true,
      weights: layer.weights,
      bias: layer.bias,
    });
    channels = outputs;
    widest = Math.max(widest, outputs);
  });

  return {
    arch: typeof o.arch === 'string' ? o.arch : 'unknown',
    scale,
    channels: isChannelCount(o.channels) ? o.channels : widest,
    luma,
    layers: parsed,
  };
}

export function convLayers(model: LinearModel): Conv3x3Layer[] {
  return model.layers.filter((layer): layer is Conv3x3Layer => layer.type === 'conv3x3');
}

/**
 * The same network in the graph format: luma in, one `conv` node per layer (3x3, replicate
 * padding, `relu` or none), the depth-to-space, and a `luma_residual` output added to the bilinear
 * upscale. The GPU runs linear models this way, through the graph's register-blocked convolution.
 */
export function linearToGraph(model: LinearModel): GraphModel {
  const nodes: GraphModel['nodes'] = [];
  let previous = GRAPH_INPUT;
  convLayers(model).forEach((layer, index) => {
    const name = `conv${index}`;
    nodes.push({
      name,
      op: 'conv',
      inputs: [previous],
      in: layer.in,
      out: layer.out,
      kernel: 3,
      dilation: 1,
      padding: 'replicate',
      activation: layer.relu ? 'relu' : 'none',
      slope: 0.01,
      weights: Float32Array.from(layer.weights),
      bias: Float32Array.from(layer.bias),
    });
    previous = name;
  });
  nodes.push({ name: 'shuffle', op: 'depth_to_space', inputs: [previous], scale: model.scale });
  return {
    arch: model.arch,
    scale: model.scale,
    channels: model.channels,
    input: 'luma',
    output: 'luma_residual',
    luma: model.luma,
    nodes,
  };
}

// ---- CPU reference ------------------------------------------------------------------------------

/** One plane per channel, `width * height` values each. */
export type Planes = Float32Array[];

/** A 3x3 convolution padded by clamping to the edge (as trained), straight from the PyTorch layout. */
export function convReference(layer: Conv3x3Layer, input: Planes, width: number, height: number) {
  const output: Planes = [];
  for (let o = 0; o < layer.out; o++) {
    const plane = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sum = layer.bias[o];
        for (let i = 0; i < layer.in; i++) {
          for (let ky = 0; ky < 3; ky++) {
            const sy = Math.min(height - 1, Math.max(0, y + ky - 1));
            for (let kx = 0; kx < 3; kx++) {
              const sx = Math.min(width - 1, Math.max(0, x + kx - 1));
              sum +=
                layer.weights[((o * layer.in + i) * 3 + ky) * 3 + kx] * input[i][sy * width + sx];
            }
          }
        }
        plane[y * width + x] = layer.relu ? Math.max(0, sum) : sum;
      }
    }
    output.push(plane);
  }
  return output;
}

/** `nn.PixelShuffle(scale)` for a single output channel. */
export function depthToSpaceReference(
  input: Planes,
  scale: number,
  width: number,
  height: number,
): Float32Array {
  const outWidth = width * scale;
  const out = new Float32Array(outWidth * height * scale);
  for (let y = 0; y < height * scale; y++) {
    for (let x = 0; x < outWidth; x++) {
      const channel = (y % scale) * scale + (x % scale);
      out[y * outWidth + x] = input[channel][Math.floor(y / scale) * width + Math.floor(x / scale)];
    }
  }
  return out;
}

/** The luma residual the network predicts for a luma plane: `width*scale x height*scale` values. */
export function runModelReference(
  model: LinearModel,
  luma: Float32Array,
  width: number,
  height: number,
): Float32Array {
  let planes: Planes = [luma];
  for (const layer of convLayers(model)) planes = convReference(layer, planes, width, height);
  return depthToSpaceReference(planes, model.scale, width, height);
}

/**
 * The bilinear base the residual is added to: `F.interpolate(mode='bilinear',
 * align_corners=False)`, pixel centres at half-pixel offsets, clamped to the edge. One plane in,
 * one plane `scale` times the size out.
 */
export function bilinearReference(
  plane: Float32Array,
  width: number,
  height: number,
  scale: number,
): Float32Array {
  const outWidth = width * scale;
  const outHeight = height * scale;
  const out = new Float32Array(outWidth * outHeight);
  const at = (x: number, y: number) => plane[y * width + x];
  for (let y = 0; y < outHeight; y++) {
    const fy = Math.min(height - 1, Math.max(0, (y + 0.5) / scale - 0.5));
    const y0 = Math.floor(fy);
    const y1 = Math.min(height - 1, y0 + 1);
    const ty = fy - y0;
    for (let x = 0; x < outWidth; x++) {
      const fx = Math.min(width - 1, Math.max(0, (x + 0.5) / scale - 0.5));
      const x0 = Math.floor(fx);
      const x1 = Math.min(width - 1, x0 + 1);
      const tx = fx - x0;
      const top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * tx;
      const bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * tx;
      out[y * outWidth + x] = top + (bottom - top) * ty;
    }
  }
  return out;
}

// ---- loading ------------------------------------------------------------------------------------

/** Weights as an object, a URL to fetch a JSON file from, or a function that resolves to one. */
export type ModelSource = ModelWeights | object | string | URL | (() => Promise<unknown> | unknown);

type FetchLike = (
  url: string,
) => Promise<{ ok: boolean; status: number; json(): Promise<unknown> }>;

/** Loads and validates weights. Rejects with the reason they cannot be used. */
export async function loadModel(
  source: ModelSource,
  fetchJson: FetchLike = (url) => fetch(url),
): Promise<ModelWeights> {
  let json: unknown;
  if (typeof source === 'function') {
    json = await (source as () => unknown)();
  } else if (typeof source === 'string' || source instanceof URL) {
    const response = await fetchJson(String(source));
    if (!response.ok) throw new ModelError(`fetching weights failed: HTTP ${response.status}`);
    json = await response.json();
  } else {
    json = source;
  }
  return parseModel(json);
}
