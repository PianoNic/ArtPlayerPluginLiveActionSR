/**
 * The luma CNN behind `balanced` and `quality`: the weights format, its validation, how the
 * weights are packed for the WGSL convolution, and a CPU reference of the whole network.
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

/** Luma coefficients the network was trained with. */
export type LumaCoefficients = 'bt709' | 'bt601';

export interface ModelWeights {
  arch: string;
  scale: number;
  /** Width of the hidden layers; informational. */
  channels: number;
  /** Default `bt709`. */
  luma: LumaCoefficients;
  layers: ModelLayer[];
}

/** Hard limits, so a broken file cannot generate a shader that takes minutes to compile. */
export const MAX_CHANNELS = 64;
export const MAX_LAYERS = 32;
export const SUPPORTED_SCALES: readonly number[] = [2, 3, 4];

export const LUMA_WEIGHTS: Record<LumaCoefficients, readonly [number, number, number]> = {
  bt709: [0.2126, 0.7152, 0.0722],
  bt601: [0.299, 0.587, 0.114],
};

export class ModelError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ModelError';
  }
}

function isFiniteArray(value: unknown, length: number): value is number[] {
  return (
    Array.isArray(value) &&
    value.length === length &&
    value.every((n) => typeof n === 'number' && Number.isFinite(n))
  );
}

function isChannelCount(value: unknown): value is number {
  return (
    typeof value === 'number' && Number.isInteger(value) && value >= 1 && value <= MAX_CHANNELS
  );
}

/** Checks a parsed weights file and returns it typed. Throws `ModelError` naming the first problem. */
export function parseModel(json: unknown): ModelWeights {
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

export function convLayers(model: ModelWeights): Conv3x3Layer[] {
  return model.layers.filter((layer): layer is Conv3x3Layer => layer.type === 'conv3x3');
}

/** Channels travel in groups of four: one RGBA texel of a texture array layer per group. */
export function groups(channels: number): number {
  return Math.ceil(channels / 4);
}

/**
 * Packs a layer's weights for the WGSL convolution: one `mat4x4` per output group, input group and
 * tap, laid out `[og][ig][tap]`. Column `k` of a matrix holds the weights from input channel
 * `ig*4 + k` to output channels `og*4 .. og*4 + 3`, so `matrix * inputTexel` is that tap's
 * contribution to four outputs. Taps run row by row, `tap = ky*3 + kx`. Missing channels are 0.
 */
export function packConvWeights(layer: Conv3x3Layer): Float32Array<ArrayBuffer> {
  const ig = groups(layer.in);
  const og = groups(layer.out);
  const packed = new Float32Array(og * ig * 9 * 16);
  for (let o = 0; o < layer.out; o++) {
    for (let i = 0; i < layer.in; i++) {
      for (let t = 0; t < 9; t++) {
        const matrix = ((o >> 2) * ig + (i >> 2)) * 9 + t;
        packed[matrix * 16 + (i & 3) * 4 + (o & 3)] = layer.weights[(o * layer.in + i) * 9 + t];
      }
    }
  }
  return packed;
}

/** The bias as one `vec4` per output group, zero-padded. */
export function packBias(layer: Conv3x3Layer): Float32Array<ArrayBuffer> {
  const packed = new Float32Array(groups(layer.out) * 4);
  packed.set(layer.bias);
  return packed;
}

const scratch = new DataView(new ArrayBuffer(4));

/** IEEE half-precision bits of a number, rounded to nearest even; overflow becomes infinity. */
export function toHalfBits(value: number): number {
  scratch.setFloat32(0, value);
  const x = scratch.getUint32(0);
  const sign = (x >>> 16) & 0x8000;
  const exponent = (x >>> 23) & 0xff;
  let mantissa = x & 0x7fffff;
  if (exponent === 0xff) return sign | 0x7c00 | (mantissa ? 0x200 : 0);
  const e = exponent - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    mantissa |= 0x800000;
    const shift = 14 - e;
    let half = mantissa >>> shift;
    const rest = mantissa & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    if (rest > halfway || (rest === halfway && half & 1)) half++;
    return sign | half;
  }
  let half = (e << 10) | (mantissa >>> 13);
  const rest = mantissa & 0x1fff;
  // A carry out of the mantissa rounds into the exponent, up to infinity, as it should.
  if (rest > 0x1000 || (rest === 0x1000 && half & 1)) half++;
  return sign | half;
}

/** The value a half-precision bit pattern stands for. */
export function fromHalfBits(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 0x1f) return mantissa ? NaN : sign * Infinity;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

/** Packed weights as half floats, for the `shader-f16` path. */
export function toHalfArray(values: Float32Array): Uint16Array<ArrayBuffer> {
  const out = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = toHalfBits(values[i]);
  return out;
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

/**
 * The same convolution computed the way the WGSL shader does it: from packed matrices and
 * channel groups. Used to prove the packing is right.
 */
export function convPackedReference(
  packed: Float32Array,
  bias: Float32Array,
  inputGroups: number,
  outputGroups: number,
  relu: boolean,
  input: Planes,
  width: number,
  height: number,
): Planes {
  const texel = (channel: number, x: number, y: number) =>
    channel < input.length ? input[channel][y * width + x] : 0;
  const output: Planes = Array.from(
    { length: outputGroups * 4 },
    () => new Float32Array(width * height),
  );
  for (let og = 0; og < outputGroups; og++) {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const acc = [bias[og * 4], bias[og * 4 + 1], bias[og * 4 + 2], bias[og * 4 + 3]];
        for (let t = 0; t < 9; t++) {
          const qx = Math.min(width - 1, Math.max(0, x + (t % 3) - 1));
          const qy = Math.min(height - 1, Math.max(0, y + Math.floor(t / 3) - 1));
          for (let ig = 0; ig < inputGroups; ig++) {
            const m = ((og * inputGroups + ig) * 9 + t) * 16;
            for (let k = 0; k < 4; k++) {
              const a = texel(ig * 4 + k, qx, qy);
              for (let j = 0; j < 4; j++) acc[j] += packed[m + k * 4 + j] * a;
            }
          }
        }
        for (let j = 0; j < 4; j++) {
          output[og * 4 + j][y * width + x] = relu ? Math.max(0, acc[j]) : acc[j];
        }
      }
    }
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
  model: ModelWeights,
  luma: Float32Array,
  width: number,
  height: number,
): Float32Array {
  let planes: Planes = [luma];
  for (const layer of convLayers(model)) planes = convReference(layer, planes, width, height);
  return depthToSpaceReference(planes, model.scale, width, height);
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
