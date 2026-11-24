import { HASH_WGSL, LUMA_WGSL } from './common.wgsl.js';

/**
 * Draws the processed frame onto the canvas and adds film grain on the way. Written for this
 * project (MIT).
 *
 * Grain comes last and at the output resolution: grain added before an upscale is upscaled with
 * the picture and turns blotchy. It is luma-only (the same offset on R, G and B), triangular
 * noise from two hashes of the pixel and the frame number, so it moves like film grain. It is
 * strongest in the midtones and fades towards black and white, where real grain is least visible
 * and where an offset would clip. Besides looking like film, it masks what deband smoothed and the
 * banding of the 8-bit canvas.
 */
export const PRESENT_WGSL = /* wgsl */ `
${HASH_WGSL}
${LUMA_WGSL}

struct Params {
  grain : f32,
  frame : u32,
  _pad0 : u32,
  _pad1 : u32,
}

struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) uv : vec2f,
}

@group(0) @binding(0) var frameSampler : sampler;
@group(0) @binding(1) var frameTexture : texture_2d<f32>;
@group(0) @binding(2) var<uniform> params : Params;

// One triangle that covers the whole target.
@vertex
fn vert_main(@builtin(vertex_index) index : u32) -> VertexOutput {
  let xy = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  var out : VertexOutput;
  out.position = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2f(xy.x, 1.0 - xy.y);
  return out;
}

@fragment
fn frag_main(v : VertexOutput) -> @location(0) vec4f {
  var color = textureSampleLevel(frameTexture, frameSampler, v.uv, 0.0).rgb;
  if (params.grain > 0.0) {
    let p = vec2u(v.position.xy);
    let h = hash3(vec3u(p, params.frame));
    let noise = rand01(h) + rand01(pcg(h)) - 1.0;
    let y = clamp(luma709(color), 0.0, 1.0);
    let mask = 0.2 + 3.2 * y * (1.0 - y);
    color += vec3f(noise * params.grain * mask);
  }
  return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
}
`;
