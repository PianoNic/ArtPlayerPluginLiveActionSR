/**
 * Helpers shared by several shaders. Written for this project (MIT).
 *
 * `pcg` is the PCG-based integer hash evaluated in Jarzynski & Olano, "Hash Functions for GPU
 * Rendering" (JCGT 2020): one well-mixed 32-bit value per input, no state, no textures.
 */
export const HASH_WGSL = /* wgsl */ `
fn pcg(v : u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn hash3(p : vec3u) -> u32 {
  return pcg(p.x ^ pcg(p.y ^ pcg(p.z)));
}

// Uniform in [0, 1).
fn rand01(h : u32) -> f32 {
  return f32(h >> 8u) * (1.0 / 16777216.0);
}
`;

/** Rec. 709 luma of a non-linear RGB value. */
export const LUMA_WGSL = /* wgsl */ `
fn luma709(c : vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}
`;
