import { LUMA_WEIGHTS, type LumaCoefficients } from '../model.js';

/**
 * The luma CNN, as three kinds of compute passes. Written for this project (MIT).
 *
 * Activations live in `rgba16float` texture arrays: layer `g` of the array holds channels
 * `4g .. 4g+3`. Every convolution is one dispatch with one invocation per pixel and output group;
 * the weights come packed as `mat4x4` per output group, input group and tap (see
 * `packConvWeights`). With `shader-f16` the arithmetic and the weights are half precision.
 */

function floatType(f16: boolean): string {
  return f16 ? 'f16' : 'f32';
}

/** Writes the source luma into layer 0 of a texture array: `vec4(Y, 0, 0, 0)`. */
export function lumaWgsl(coefficients: LumaCoefficients): string {
  const [r, g, b] = LUMA_WEIGHTS[coefficients];
  return /* wgsl */ `
@group(0) @binding(0) var src : texture_2d<f32>;
@group(0) @binding(1) var dst : texture_storage_2d_array<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id : vec3u) {
  let size = textureDimensions(src);
  if (id.x >= size.x || id.y >= size.y) {
    return;
  }
  let c = textureLoad(src, vec2i(id.xy), 0).rgb;
  let y = dot(c, vec3f(${r}, ${g}, ${b}));
  textureStore(dst, vec2i(id.xy), 0, vec4f(y, 0.0, 0.0, 0.0));
}
`;
}

/**
 * A 3x3 convolution from `inputGroups` to `outputGroups` channel groups. The border is padded by
 * clamping to the edge (replicate), as the training does (training/README.md): zero padding would
 * darken the frame edges.
 */
export function convWgsl(
  inputGroups: number,
  outputGroups: number,
  relu: boolean,
  f16: boolean,
): string {
  const F = floatType(f16);
  return /* wgsl */ `
${f16 ? 'enable f16;' : ''}

@group(0) @binding(0) var src : texture_2d_array<f32>;
@group(0) @binding(1) var dst : texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(2) var<storage, read> weights : array<mat4x4<${F}>>;
@group(0) @binding(3) var<storage, read> bias : array<vec4<${F}>>;

const INPUT_GROUPS : u32 = ${inputGroups}u;
const OUTPUT_GROUPS : u32 = ${outputGroups}u;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id : vec3u) {
  let size = textureDimensions(dst);
  let og = id.z;
  if (id.x >= size.x || id.y >= size.y || og >= OUTPUT_GROUPS) {
    return;
  }
  let p = vec2i(id.xy);
  let maxP = vec2i(size) - 1;
  var acc = bias[og];
  for (var t = 0u; t < 9u; t++) {
    let q = clamp(p + vec2i(i32(t % 3u) - 1, i32(t / 3u) - 1), vec2i(0), maxP);
    for (var ig = 0u; ig < INPUT_GROUPS; ig++) {
      let a = vec4<${F}>(textureLoad(src, q, ig, 0));
      acc += weights[(og * INPUT_GROUPS + ig) * 9u + t] * a;
    }
  }
  ${relu ? `acc = max(acc, vec4<${F}>(0.0));` : ''}
  textureStore(dst, p, og, vec4f(acc));
}
`;
}

/**
 * Depth-to-space plus the residual: each output pixel takes its channel of the last convolution
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
