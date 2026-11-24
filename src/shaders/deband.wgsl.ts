import { HASH_WGSL, LUMA_WGSL } from './common.wgsl.js';

/**
 * Deband, at the source resolution. Written from scratch for this project (MIT) after the
 * published description of f3kdb-style debanding; no code from libplacebo, mpv or f3kdb.
 *
 * Banding is a staircase of nearly equal values with long flat steps. For each pixel and each
 * iteration, four neighbours are read on a cross at a pseudo-random angle and distance; their mean
 * replaces the pixel when it lies within a threshold of it. Within the threshold the pixel is on a
 * gentle gradient (or a band edge), and averaging over a wide random footprint turns the step into a
 * ramp. Beyond it the neighbours belong to real detail and the pixel is left alone. The blend fades
 * out smoothly towards the threshold, so detail that sits right at it does not flicker in and out.
 * Later iterations look closer with a tighter threshold, refining what the first one smoothed.
 *
 * The random offsets depend only on the pixel position, so the pattern is static from frame to
 * frame; the grain stage adds the moving noise that hides what is left.
 */
export const DEBAND_WGSL = /* wgsl */ `
${HASH_WGSL}
${LUMA_WGSL}

struct Params {
  threshold : f32,
  radius : f32,
  iterations : u32,
  _pad : u32,
}

@group(0) @binding(0) var src : texture_2d<f32>;
@group(0) @binding(1) var dst : texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params : Params;

fn fetch(p : vec2i, maxP : vec2i) -> vec3f {
  return textureLoad(src, clamp(p, vec2i(0), maxP), 0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id : vec3u) {
  let size = textureDimensions(src);
  if (id.x >= size.x || id.y >= size.y) {
    return;
  }
  let p = vec2i(id.xy);
  let maxP = vec2i(size) - 1;
  let original = fetch(p, maxP);
  var color = original;

  for (var i = 1u; i <= params.iterations; i++) {
    let h = hash3(vec3u(id.xy, i));
    let r1 = rand01(h);
    let r2 = rand01(pcg(h));
    let reach = params.radius * f32(params.iterations - i + 1u) / f32(params.iterations);
    let dist = max(1.0, reach * r1);
    let angle = r2 * 6.28318530718;
    let o = vec2i(round(vec2f(cos(angle), sin(angle)) * dist));
    let q = vec2i(-o.y, o.x);
    let mean = 0.25 * (fetch(p + o, maxP) + fetch(p - o, maxP) + fetch(p + q, maxP) + fetch(p - q, maxP));

    // Judge on the larger of the luma and the per-channel difference: banding in a flat sky
    // differs in all three, a coloured edge may only differ in one.
    let diff = abs(mean - original);
    let d = max(abs(luma709(mean) - luma709(original)), max(diff.r, max(diff.g, diff.b)));
    let threshold = params.threshold / f32(i);
    let w = clamp(1.0 - d / max(threshold, 1e-6), 0.0, 1.0);
    color = mix(color, mean, w * w);
  }

  textureStore(dst, p, vec4f(color, 1.0));
}
`;
