/**
 * AMD FidelityFX Super Resolution 1.0 (FSR 1): EASU (edge-adaptive spatial upsampling) and RCAS
 * (robust contrast-adaptive sharpening), ported to WGSL from AMD's `ffx_fsr1.h`.
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 * and associated documentation files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Differences from the original: the 32-bit path only; the 12 taps are read with `textureLoad`
 * (clamped to the edge) instead of four gathers; AMD's bit-trick reciprocal approximations are
 * plain divisions, guarded against zero; RCAS runs with its noise-detection term
 * (`FSR_RCAS_DENOISE`) always on, which suits compressed live-action video.
 */

/** EASU: one output pixel per invocation; the input is the texture at binding 0. */
export const EASU_WGSL = /* wgsl */ `
@group(0) @binding(0) var src : texture_2d<f32>;
@group(0) @binding(1) var dst : texture_storage_2d<rgba16float, write>;

var<private> maxP : vec2i;

fn tap(p : vec2i) -> vec3f {
  return textureLoad(src, clamp(p, vec2i(0), maxP), 0).rgb;
}

// Luma times two, as FSR computes it: B/2 + R/2 + G.
fn lum2(c : vec3f) -> f32 {
  return c.b * 0.5 + (c.r * 0.5 + c.g);
}

// Accumulates direction and edge length for one of the four bilinear corners.
//      a
//    b c d
//      e
fn easuSet(w : f32, lA : f32, lB : f32, lC : f32, lD : f32, lE : f32, dir : ptr<function, vec2f>, len : ptr<function, f32>) {
  let dc = lD - lC;
  let cb = lC - lB;
  let dirX = lD - lB;
  var lenX = abs(dirX) / max(max(abs(dc), abs(cb)), 1e-8);
  lenX = clamp(lenX, 0.0, 1.0);
  lenX *= lenX;

  let ec = lE - lC;
  let ca = lC - lA;
  let dirY = lE - lA;
  var lenY = abs(dirY) / max(max(abs(ec), abs(ca)), 1e-8);
  lenY = clamp(lenY, 0.0, 1.0);
  lenY *= lenY;

  *dir += vec2f(dirX, dirY) * w;
  *len += (lenX + lenY) * w;
}

// One filter tap: an approximated Lanczos-2 stretched along the edge.
fn easuTap(aC : ptr<function, vec3f>, aW : ptr<function, f32>, off : vec2f, dir : vec2f, len : vec2f, lob : f32, clp : f32, c : vec3f) {
  var v = vec2f(off.x * dir.x + off.y * dir.y, off.x * -dir.y + off.y * dir.x);
  v *= len;
  let d2 = min(v.x * v.x + v.y * v.y, clp);
  // (25/16 * (2/5 * x^2 - 1)^2 - (25/16 - 1)) * (lob * x^2 - 1)^2
  var wB = 2.0 / 5.0 * d2 - 1.0;
  var wA = lob * d2 - 1.0;
  wB *= wB;
  wA *= wA;
  wB = 25.0 / 16.0 * wB - (25.0 / 16.0 - 1.0);
  let w = wB * wA;
  *aC += c * w;
  *aW += w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id : vec3u) {
  let outSize = textureDimensions(dst);
  if (id.x >= outSize.x || id.y >= outSize.y) {
    return;
  }
  let inSize = vec2f(textureDimensions(src));
  maxP = vec2i(textureDimensions(src)) - 1;

  // Position of 'f' and the fraction past it.
  var pp = (vec2f(id.xy) + 0.5) * inSize / vec2f(outSize) - 0.5;
  let fp = floor(pp);
  pp -= fp;
  let f0 = vec2i(fp);

  // 12-tap kernel.
  //    b c
  //  e f g h
  //  i j k l
  //    n o
  let b = tap(f0 + vec2i(0, -1));
  let c = tap(f0 + vec2i(1, -1));
  let e = tap(f0 + vec2i(-1, 0));
  let f = tap(f0);
  let g = tap(f0 + vec2i(1, 0));
  let h = tap(f0 + vec2i(2, 0));
  let i = tap(f0 + vec2i(-1, 1));
  let j = tap(f0 + vec2i(0, 1));
  let k = tap(f0 + vec2i(1, 1));
  let l = tap(f0 + vec2i(2, 1));
  let n = tap(f0 + vec2i(0, 2));
  let o = tap(f0 + vec2i(1, 2));

  let bL = lum2(b);
  let cL = lum2(c);
  let eL = lum2(e);
  let fL = lum2(f);
  let gL = lum2(g);
  let hL = lum2(h);
  let iL = lum2(i);
  let jL = lum2(j);
  let kL = lum2(k);
  let lL = lum2(l);
  let nL = lum2(n);
  let oL = lum2(o);

  // Direction and length, bilinearly weighted over the four centre texels.
  var dir = vec2f(0.0);
  var len = 0.0;
  easuSet((1.0 - pp.x) * (1.0 - pp.y), bL, eL, fL, gL, jL, &dir, &len);
  easuSet(pp.x * (1.0 - pp.y), cL, fL, gL, hL, kL, &dir, &len);
  easuSet((1.0 - pp.x) * pp.y, fL, iL, jL, kL, nL, &dir, &len);
  easuSet(pp.x * pp.y, gL, jL, kL, lL, oL, &dir, &len);

  // Normalize, with a fallback for no gradient at all.
  let dir2 = dir * dir;
  var dirR = dir2.x + dir2.y;
  let zero = dirR < 1.0 / 32768.0;
  dirR = select(inverseSqrt(dirR), 1.0, zero);
  dir.x = select(dir.x, 1.0, zero);
  dir *= dirR;

  // {0 to 2} to {0 to 1}, shaped with a square.
  len = len * 0.5;
  len *= len;

  // Stretch the kernel from 1.0 (horizontal or vertical) to sqrt(2.0) (diagonal).
  let stretch = (dir.x * dir.x + dir.y * dir.y) / max(max(abs(dir.x), abs(dir.y)), 1e-8);
  let len2 = vec2f(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);
  // The window moves from +/- sqrt(2.0) to slightly beyond 2.0 with the amount of edge.
  let lob = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * len;
  let clp = 1.0 / lob;

  // De-ringing limits: the four nearest texels.
  let min4 = min(min(f, g), min(j, k));
  let max4 = max(max(f, g), max(j, k));

  var aC = vec3f(0.0);
  var aW = 0.0;
  easuTap(&aC, &aW, vec2f(0.0, -1.0) - pp, dir, len2, lob, clp, b);
  easuTap(&aC, &aW, vec2f(1.0, -1.0) - pp, dir, len2, lob, clp, c);
  easuTap(&aC, &aW, vec2f(-1.0, 1.0) - pp, dir, len2, lob, clp, i);
  easuTap(&aC, &aW, vec2f(0.0, 1.0) - pp, dir, len2, lob, clp, j);
  easuTap(&aC, &aW, vec2f(0.0, 0.0) - pp, dir, len2, lob, clp, f);
  easuTap(&aC, &aW, vec2f(-1.0, 0.0) - pp, dir, len2, lob, clp, e);
  easuTap(&aC, &aW, vec2f(1.0, 1.0) - pp, dir, len2, lob, clp, k);
  easuTap(&aC, &aW, vec2f(2.0, 1.0) - pp, dir, len2, lob, clp, l);
  easuTap(&aC, &aW, vec2f(2.0, 0.0) - pp, dir, len2, lob, clp, h);
  easuTap(&aC, &aW, vec2f(1.0, 0.0) - pp, dir, len2, lob, clp, g);
  easuTap(&aC, &aW, vec2f(1.0, 2.0) - pp, dir, len2, lob, clp, o);
  easuTap(&aC, &aW, vec2f(0.0, 2.0) - pp, dir, len2, lob, clp, n);

  let pix = min(max4, max(min4, aC / aW));
  textureStore(dst, vec2i(id.xy), vec4f(pix, 1.0));
}
`;

/** RCAS: sharpens in place at one resolution; `sharpness` is the linear lobe multiplier. */
export const RCAS_WGSL = /* wgsl */ `
struct Params {
  sharpness : f32,
  _pad0 : f32,
  _pad1 : f32,
  _pad2 : f32,
}

@group(0) @binding(0) var src : texture_2d<f32>;
@group(0) @binding(1) var dst : texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params : Params;

// The lobe limit that keeps the resolve from going negative: 0.25 - 1/16.
const RCAS_LIMIT : f32 = 0.25 - 1.0 / 16.0;

fn max3(a : vec3f, b : vec3f, c : vec3f) -> vec3f {
  return max(a, max(b, c));
}

fn min3(a : vec3f, b : vec3f, c : vec3f) -> vec3f {
  return min(a, min(b, c));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id : vec3u) {
  let size = textureDimensions(src);
  if (id.x >= size.x || id.y >= size.y) {
    return;
  }
  let sp = vec2i(id.xy);
  let maxP = vec2i(size) - 1;

  //    b
  //  d e f
  //    h
  let b = textureLoad(src, clamp(sp + vec2i(0, -1), vec2i(0), maxP), 0).rgb;
  let d = textureLoad(src, clamp(sp + vec2i(-1, 0), vec2i(0), maxP), 0).rgb;
  let e = textureLoad(src, sp, 0).rgb;
  let f = textureLoad(src, clamp(sp + vec2i(1, 0), vec2i(0), maxP), 0).rgb;
  let h = textureLoad(src, clamp(sp + vec2i(0, 1), vec2i(0), maxP), 0).rgb;

  // Luma times two.
  let bL = b.b * 0.5 + (b.r * 0.5 + b.g);
  let dL = d.b * 0.5 + (d.r * 0.5 + d.g);
  let eL = e.b * 0.5 + (e.r * 0.5 + e.g);
  let fL = f.b * 0.5 + (f.r * 0.5 + f.g);
  let hL = h.b * 0.5 + (h.r * 0.5 + h.g);

  // Noise detection: a lone peak against a quiet ring is noise, and gets half the sharpening.
  var nz = 0.25 * bL + 0.25 * dL + 0.25 * fL + 0.25 * hL - eL;
  let range = max(max(max(bL, dL), max(eL, fL)), hL) - min(min(min(bL, dL), min(eL, fL)), hL);
  nz = clamp(abs(nz) / max(range, 1e-8), 0.0, 1.0);
  nz = -0.5 * nz + 1.0;

  // Min and max of the ring.
  let mn4 = min(min3(b, d, f), h);
  let mx4 = max(max3(b, d, f), h);

  // Limiters: the largest negative lobe that keeps the result within 0..1.
  let hitMin = min(mn4, e) / max(4.0 * mx4, vec3f(1e-8));
  let hitMax = (1.0 - max(mx4, e)) / min(4.0 * mn4 - 4.0, vec3f(-1e-8));
  let lobeRGB = max(-hitMin, hitMax);
  var lobe = max(-RCAS_LIMIT, min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), 0.0)) * params.sharpness;
  lobe *= nz;

  let pix = (lobe * (b + d + h + f) + e) / (4.0 * lobe + 1.0);
  textureStore(dst, sp, vec4f(pix, 1.0));
}
`;
