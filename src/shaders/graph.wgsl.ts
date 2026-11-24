import type { Activation, GraphInput, Padding } from '../graph.js';
import { LUMA_WEIGHTS, type LumaCoefficients } from '../tensor.js';

/** Storage format of the activation maps. */
export type ActivationFormat = 'rgba16float' | 'rgba32float';

/**
 * The passes of a graph network (see `graph.ts`). Written for this project (MIT).
 *
 * Maps live in `rgba16float` (or `rgba32float`) texture arrays: layer `g` holds channels
 * `4g .. 4g+3`. Vectors (per-channel values, from a global pool) live in small `f32`
 * storage buffers, zero-padded to whole groups. Activation functions are evaluated in `f32` even on
 * the `shader-f16` path: a half-precision `exp` overflows long before `tanh` saturates.
 */

/** WGSL for `act(v)` on a `vec4f`. */
function activationWgsl(activation: Activation, slope: number): string {
  switch (activation) {
    case 'relu':
      return 'max(v, vec4f(0.0))';
    case 'lrelu':
      return `select(v * ${floatLiteral(slope)}, v, v >= vec4f(0.0))`;
    case 'sigmoid':
      return '1.0 / (1.0 + exp(-clamp(v, vec4f(-30.0), vec4f(30.0))))';
    case 'tanh':
      return 'tanh(clamp(v, vec4f(-15.0), vec4f(15.0)))';
    default:
      return 'v';
  }
}

/** A finite number as a WGSL `f32` literal. */
function floatLiteral(value: number): string {
  const text = String(value);
  return /[.eE]/.test(text) ? text : `${text}.0`;
}

function activationFn(activation: Activation, slope: number): string {
  return /* wgsl */ `
fn act(v : vec4f) -> vec4f {
  return ${activationWgsl(activation, slope)};
}
`;
}

/** The frame into layer 0 of a texture array: `vec4(R, G, B, 0)`, or `vec4(Y', 0, 0, 0)`. */
export function graphInputWgsl(
  input: GraphInput,
  coefficients: LumaCoefficients,
  format: ActivationFormat,
): string {
  const [r, g, b] = LUMA_WEIGHTS[coefficients];
  const value =
    input === 'rgb' ? 'vec4f(c, 0.0)' : `vec4f(dot(c, vec3f(${r}, ${g}, ${b})), 0.0, 0.0, 0.0)`;
  return /* wgsl */ `
@group(0) @binding(0) var src : texture_2d<f32>;
@group(0) @binding(1) var dst : texture_storage_2d_array<${format}, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id : vec3u) {
  let size = textureDimensions(src);
  if (id.x >= size.x || id.y >= size.y) {
    return;
  }
  let c = textureLoad(src, vec2i(id.xy), 0).rgb;
  textureStore(dst, vec2i(id.xy), 0, ${value});
}
`;
}

export interface ConvShader {
  inputGroups: number;
  /** Output groups after padding: a multiple of `perInvocation`. */
  outputGroups: number;
  /** Real output groups; the padded ones are computed (from zero weights) but not stored. */
  storedGroups: number;
  /** Output groups one invocation computes: every input texel it loads serves all of them. */
  perInvocation: number;
  /** Horizontally adjacent pixels one invocation computes: every weight it loads serves all of them. */
  pixels: number;
  kernel: number;
  dilation: number;
  padding: Padding;
  activation: Activation;
  slope: number;
  /** Half-precision weights and products (`shader-f16`). */
  f16: boolean;
  /** Accumulate in `f32` even with `f16` products. */
  accumulateF32: boolean;
  format: ActivationFormat;
}

/**
 * A same-size KxK convolution, register-blocked: one invocation computes `pixels` horizontally
 * adjacent pixels for `perInvocation` output groups (`global_invocation_id.z` picks the chunk of
 * groups). For each kernel row and input group it loads the row's texels once, then every weight
 * it reads is applied to all its pixels; the weight reads are uniform across the workgroup. Zero
 * padding reads 0 outside the frame; replicate padding clamps to the edge.
 */
export function graphConvWgsl(s: ConvShader): string {
  const W = s.f16 ? 'f16' : 'f32';
  const A = s.f16 && !s.accumulateF32 ? 'f16' : 'f32';
  const K = s.kernel;
  const D = s.dilation;
  const P = s.pixels;
  const taps = K * K;
  const radius = (D * (K - 1)) / 2;
  const per = Array.from({ length: s.perInvocation }, (_, j) => j);
  const pix = Array.from({ length: P }, (_, p) => p);
  const used = new Set<number>();
  for (let kx = 0; kx < K; kx++) for (const p of pix) used.add(p + kx * D);
  const texels = [...used].sort((a, b) => a - b);
  const zero = s.padding === 'zero';
  const load = (n: number) => {
    const x = `x0 + ${n - radius}`;
    return zero
      ? `    let t${n} = vec4<${W}>(select(vec4f(0.0), textureLoad(src, vec2i(clamp(${x}, 0, maxP.x), y), ig, 0), ${x} >= 0 && ${x} <= maxP.x));`
      : `    let t${n} = vec4<${W}>(textureLoad(src, vec2i(clamp(${x}, 0, maxP.x), y), ig, 0));`;
  };
  const body: string[] = [];
  for (let kx = 0; kx < K; kx++) {
    for (const j of per) {
      body.push(`    let m${j}_${kx} = weights[w + ${j * s.inputGroups * taps + kx}u];`);
      for (const p of pix) {
        const product = `m${j}_${kx} * t${p + kx * D}`;
        body.push(`    a${j}_${p} += ${A === W ? product : `vec4<${A}>(${product})`};`);
      }
    }
  }
  return /* wgsl */ `
${s.f16 ? 'enable f16;' : ''}

@group(0) @binding(0) var src : texture_2d_array<f32>;
@group(0) @binding(1) var dst : texture_storage_2d_array<${s.format}, write>;
@group(0) @binding(2) var<storage, read> weights : array<mat4x4<${W}>>;
@group(0) @binding(3) var<storage, read> bias : array<vec4<${W}>>;

const INPUT_GROUPS : u32 = ${s.inputGroups}u;
const STORED_GROUPS : u32 = ${s.storedGroups}u;
const PER : u32 = ${s.perInvocation}u;
const PIXELS : u32 = ${P}u;
const KERNEL : u32 = ${K}u;
const TAPS : u32 = ${taps}u;
${activationFn(s.activation, s.slope)}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id : vec3u) {
  let size = vec2i(textureDimensions(dst));
  let x0 = i32(id.x * PIXELS);
  let row = i32(id.y);
  if (x0 >= size.x || row >= size.y) {
    return;
  }
  let og = id.z * PER;
  let maxP = size - 1;
${per.map((j) => pix.map((p) => `  var a${j}_${p} = vec4<${A}>(bias[og + ${j}u]);`).join('\n')).join('\n')}
  for (var ky = 0u; ky < KERNEL; ky++) {
    let y0 = row + i32(ky) * ${D} - ${radius};
${
  zero
    ? `    if (y0 < 0 || y0 > maxP.y) {
      continue;
    }
    let y = y0;`
    : '    let y = clamp(y0, 0, maxP.y);'
}
    for (var ig = 0u; ig < INPUT_GROUPS; ig++) {
${texels.map(load).join('\n')}
    let w = (og * INPUT_GROUPS + ig) * TAPS + ky * KERNEL;
${body.join('\n')}
    }
  }
${pix
  .map(
    (p) => `  if (x0 + ${p} <= maxP.x) {
${per
  .map(
    (j) => `    if (og + ${j}u < STORED_GROUPS) {
      textureStore(dst, vec2i(x0 + ${p}, row), og + ${j}u, act(vec4f(a${j}_${p})));
    }`,
  )
  .join('\n')}
  }`,
  )
  .join('\n')}
}
`;
}

/**
 * Elementwise `add` / `mul` of two maps, or of a map and a vector (broadcast), and `activation`
 * (`op` null: one operand). One invocation per pixel and channel group.
 */
export function graphElementwiseWgsl(
  op: 'add' | 'mul' | null,
  vector: boolean,
  activation: Activation,
  slope: number,
  format: ActivationFormat,
): string {
  const second =
    op === null
      ? ''
      : vector
        ? '@group(0) @binding(2) var<storage, read> b : array<vec4f>;'
        : '@group(0) @binding(2) var b : texture_2d_array<f32>;';
  const operand = vector ? 'b[g]' : 'textureLoad(b, p, g, 0)';
  const value = op === null ? 'x' : op === 'add' ? `x + ${operand}` : `x * ${operand}`;
  return /* wgsl */ `
@group(0) @binding(0) var a : texture_2d_array<f32>;
@group(0) @binding(1) var dst : texture_storage_2d_array<${format}, write>;
${second}
${activationFn(activation, slope)}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id : vec3u) {
  let size = textureDimensions(dst);
  let g = id.z;
  if (id.x >= size.x || id.y >= size.y || g >= textureNumLayers(dst)) {
    return;
  }
  let p = vec2i(id.xy);
  let x = textureLoad(a, p, g, 0);
  textureStore(dst, p, g, act(${value}));
}
`;
}

/** Pixels per side of the tile one workgroup of the pooling pass sums. */
export const POOL_TILE = 32;

/**
 * First half of a global average pool: each 8x8 workgroup sums one 32x32 tile of one channel group
 * in `f32` and writes the sum to `partial[group * tiles + tile]`.
 */
export const GRAPH_POOL_PARTIAL_WGSL = /* wgsl */ `
@group(0) @binding(0) var src : texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read_write> partial : array<vec4f>;

const TILE : u32 = ${POOL_TILE}u;

var<workgroup> sums : array<vec4f, 64>;

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(workgroup_id) wg : vec3u,
  @builtin(local_invocation_id) lid : vec3u,
  @builtin(local_invocation_index) li : u32,
) {
  let size = textureDimensions(src);
  let tiles = (size + vec2u(TILE - 1u)) / TILE;
  var s = vec4f(0.0);
  for (var dy = 0u; dy < TILE; dy += 8u) {
    for (var dx = 0u; dx < TILE; dx += 8u) {
      let p = wg.xy * TILE + lid.xy + vec2u(dx, dy);
      if (p.x < size.x && p.y < size.y) {
        s += textureLoad(src, vec2i(p), wg.z, 0);
      }
    }
  }
  sums[li] = s;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (li < stride) {
      sums[li] += sums[li + stride];
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    partial[wg.z * tiles.x * tiles.y + wg.y * tiles.x + wg.x] = sums[0];
  }
}
`;

/**
 * Second half: one workgroup per channel group adds up that group's tile sums and divides by the
 * pixel count. `params.tiles` is the number of tiles per group, `params.scale` is 1 / pixels.
 */
export const GRAPH_POOL_FINAL_WGSL = /* wgsl */ `
struct Params {
  tiles : u32,
  scale : f32,
}

@group(0) @binding(0) var<storage, read> partial : array<vec4f>;
@group(0) @binding(1) var<storage, read_write> result : array<vec4f>;
@group(0) @binding(2) var<uniform> params : Params;

var<workgroup> sums : array<vec4f, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg : vec3u, @builtin(local_invocation_index) li : u32) {
  let n = params.tiles;
  var s = vec4f(0.0);
  for (var i = li; i < n; i += 256u) {
    s += partial[wg.x * n + i];
  }
  sums[li] = s;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (li < stride) {
      sums[li] += sums[li + stride];
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    result[wg.x] = sums[0] * params.scale;
  }
}
`;

/**
 * `channel_conv`: a 1-D convolution across the channels of a vector, zero padded (ECA). `weights`
 * holds the `kernel` taps followed by the bias. Channels past `channels` stay zero.
 */
export function graphChannelConvWgsl(
  channels: number,
  kernel: number,
  activation: Activation,
  slope: number,
): string {
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read> src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<storage, read> weights : array<f32>;

const CHANNELS : i32 = ${channels};
const KERNEL : i32 = ${kernel};

fn act1(x : f32) -> f32 {
  let v = vec4f(x);
  return (${activationWgsl(activation, slope)}).x;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id : vec3u) {
  let c = i32(id.x);
  if (c >= CHANNELS) {
    return;
  }
  var sum = weights[KERNEL];
  for (var j = 0; j < KERNEL; j++) {
    let k = c + j - KERNEL / 2;
    if (k >= 0 && k < CHANNELS) {
      sum += weights[j] * src[k];
    }
  }
  dst[c] = act1(sum);
}
`;
}

/**
 * The picture of an `rgb` network: depth-to-space of its `3 * scale^2` channels (PixelShuffle
 * order, channel `c * s^2 + dy * s + dx`), clamped to 0..1. No residual: the network outputs the
 * frame itself.
 */
export function graphRgbOutputWgsl(scale: number): string {
  return /* wgsl */ `
@group(0) @binding(0) var src : texture_2d_array<f32>;
@group(0) @binding(1) var dst : texture_storage_2d<rgba16float, write>;

const SCALE : u32 = ${scale}u;

fn channel(cell : vec2i, index : u32) -> f32 {
  return textureLoad(src, cell, i32(index / 4u), 0)[index % 4u];
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id : vec3u) {
  let size = textureDimensions(dst);
  if (id.x >= size.x || id.y >= size.y) {
    return;
  }
  let cell = vec2i(id.xy / SCALE);
  let sub = id.xy % SCALE;
  let base = sub.y * SCALE + sub.x;
  let rgb = vec3f(
    channel(cell, base),
    channel(cell, SCALE * SCALE + base),
    channel(cell, 2u * SCALE * SCALE + base),
  );
  textureStore(dst, vec2i(id.xy), vec4f(clamp(rgb, vec3f(0.0), vec3f(1.0)), 1.0));
}
`;
}

/**
 * Depth-to-space plus the residual: the output of a `luma_residual` network (and of
 * the linear format, which runs as one): each output pixel takes its channel of the last convolution
 * (`nn.PixelShuffle` order) and adds it to a bilinear upscale of the source, on R, G and B alike,
 * which moves luma by exactly that much and leaves the chroma alone.
 */
export function reconstructWgsl(scale: number): string {
  return /* wgsl */ `
@group(0) @binding(0) var src : texture_2d<f32>;
@group(0) @binding(1) var linearSampler : sampler;
@group(0) @binding(2) var residual : texture_2d_array<f32>;
@group(0) @binding(3) var dst : texture_storage_2d<rgba16float, write>;

const SCALE : u32 = ${scale}u;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id : vec3u) {
  let size = textureDimensions(dst);
  if (id.x >= size.x || id.y >= size.y) {
    return;
  }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(size);
  let base = textureSampleLevel(src, linearSampler, uv, 0.0).rgb;
  let cell = vec2i(id.xy / SCALE);
  let sub = id.xy % SCALE;
  let channel = sub.y * SCALE + sub.x;
  let texel = textureLoad(residual, cell, i32(channel / 4u), 0);
  let r = texel[channel % 4u];
  textureStore(dst, vec2i(id.xy), vec4f(clamp(base + vec3f(r), vec3f(0.0), vec3f(1.0)), 1.0));
}
`;
}
