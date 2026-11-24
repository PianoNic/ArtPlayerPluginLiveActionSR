/**
 * What both weight formats share: the error type, channel grouping, luma coefficients, half
 * floats and the encoding of weight tensors in JSON.
 */

/** Luma coefficients a luma network was trained with. */
export type LumaCoefficients = 'bt709' | 'bt601';

export const LUMA_WEIGHTS: Record<LumaCoefficients, readonly [number, number, number]> = {
  bt709: [0.2126, 0.7152, 0.0722],
  bt601: [0.299, 0.587, 0.114],
};

/** Hard limit on channels per tensor, so a broken file cannot generate an absurd shader. */
export const MAX_CHANNELS = 64;

export class ModelError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ModelError';
  }
}

/** Channels travel in groups of four: one RGBA texel of a texture array layer per group. */
export function groups(channels: number): number {
  return Math.ceil(channels / 4);
}

export function isFiniteArray(value: unknown, length: number): value is number[] {
  return (
    Array.isArray(value) &&
    value.length === length &&
    value.every((n) => typeof n === 'number' && Number.isFinite(n))
  );
}

export function isChannelCount(value: unknown): value is number {
  return (
    typeof value === 'number' && Number.isInteger(value) && value >= 1 && value <= MAX_CHANNELS
  );
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

/** A weight tensor in JSON: plain numbers, or little-endian floats in base64. */
export type EncodedTensor = number[] | { dtype: 'float16' | 'float32'; base64: string };

function base64Bytes(text: string): Uint8Array {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

/**
 * Decodes a weight tensor of `length` values. Throws `ModelError` (naming `where`) for a wrong
 * length, an unknown dtype, invalid base64 or a non-finite value.
 */
export function decodeTensor(value: unknown, length: number, where: string): Float32Array {
  if (Array.isArray(value)) {
    if (!isFiniteArray(value, length)) {
      throw new ModelError(`${where} must be ${length} finite numbers`);
    }
    return Float32Array.from(value);
  }
  if (value && typeof value === 'object') {
    const { dtype, base64 } = value as Record<string, unknown>;
    const size = dtype === 'float16' ? 2 : dtype === 'float32' ? 4 : 0;
    if (!size) throw new ModelError(`${where}: dtype must be float16 or float32`);
    if (typeof base64 !== 'string') throw new ModelError(`${where}: base64 must be a string`);
    let bytes: Uint8Array;
    try {
      bytes = base64Bytes(base64);
    } catch {
      throw new ModelError(`${where}: invalid base64`);
    }
    if (bytes.length !== length * size) {
      throw new ModelError(
        `${where} must hold ${length} ${dtype} values, got ${bytes.length} bytes`,
      );
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const out = new Float32Array(length);
    for (let i = 0; i < length; i++) {
      out[i] =
        size === 2 ? fromHalfBits(view.getUint16(i * 2, true)) : view.getFloat32(i * 4, true);
      if (!Number.isFinite(out[i])) throw new ModelError(`${where} must be finite`);
    }
    return out;
  }
  throw new ModelError(`${where} must be ${length} finite numbers`);
}
