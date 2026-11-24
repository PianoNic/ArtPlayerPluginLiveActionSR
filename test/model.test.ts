import { describe, expect, it } from 'vitest';
import {
  convPackedReference,
  convReference,
  depthToSpaceReference,
  fromHalfBits,
  groups,
  loadModel,
  ModelError,
  packBias,
  packConvWeights,
  parseLinearModel,
  parseModel,
  runModelReference,
  toHalfArray,
  toHalfBits,
  type Conv3x3Layer,
  type Planes,
} from '../src/model';

/** Deterministic pseudo-random numbers in -1..1. */
function rng(seed: number) {
  let state = seed;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 2 ** 31 - 1;
  };
}

function conv(inputs: number, outputs: number, relu: boolean, random = rng(inputs * 31 + outputs)) {
  return {
    type: 'conv3x3' as const,
    in: inputs,
    out: outputs,
    relu,
    weights: Array.from({ length: outputs * inputs * 9 }, random),
    bias: Array.from({ length: outputs }, random),
  };
}

function planes(channels: number, width: number, height: number, seed = 7): Planes {
  const random = rng(seed);
  return Array.from({ length: channels }, () =>
    Float32Array.from({ length: width * height }, random),
  );
}

function model(...layers: unknown[]) {
  return { arch: 'artcnn-like', scale: 2, channels: 8, layers };
}

const valid = () =>
  model(conv(1, 8, true), conv(8, 8, true), conv(8, 4, false), {
    type: 'depth_to_space',
    scale: 2,
  });

describe('parseModel', () => {
  it('accepts the documented format and defaults luma to bt709', () => {
    const parsed = parseLinearModel(valid());
    expect(parsed.scale).toBe(2);
    expect(parsed.luma).toBe('bt709');
    expect(parsed.layers.map((layer) => layer.type)).toEqual([
      'conv3x3',
      'conv3x3',
      'conv3x3',
      'depth_to_space',
    ]);
  });

  it('treats a missing relu as false', () => {
    const first = conv(1, 4, false) as Partial<Conv3x3Layer>;
    delete first.relu;
    const parsed = parseLinearModel(model(first, { type: 'depth_to_space', scale: 2 }));
    expect((parsed.layers[0] as Conv3x3Layer).relu).toBe(false);
  });

  const broken: [string, unknown, RegExp][] = [
    ['not an object', 'weights', /object/],
    ['a bad scale', { ...valid(), scale: 5 }, /scale/],
    ['a bad luma', { ...valid(), luma: 'srgb' }, /luma/],
    ['no depth_to_space', model(conv(1, 4, false), conv(4, 4, false)), /depth_to_space/],
    [
      'a first layer that does not take luma',
      model(conv(3, 4, false), { type: 'depth_to_space', scale: 2 }),
      /previous layer has 1/,
    ],
    [
      'layers that do not chain',
      model(conv(1, 8, true), conv(4, 4, false), { type: 'depth_to_space', scale: 2 }),
      /in is 4/,
    ],
    [
      'a last conv with the wrong channel count',
      model(conv(1, 8, true), conv(8, 3, false), { type: 'depth_to_space', scale: 2 }),
      /needs 4 input channels/,
    ],
    [
      'a depth_to_space scale that differs from the model',
      { ...model(conv(1, 4, false), { type: 'depth_to_space', scale: 3 }), scale: 2 },
      /scale must match/,
    ],
    [
      'too few weights',
      model({ ...conv(1, 4, false), weights: [1, 2, 3] }, { type: 'depth_to_space', scale: 2 }),
      /36 finite numbers/,
    ],
    [
      'a NaN weight',
      model(
        { ...conv(1, 4, false), weights: Array.from({ length: 36 }, () => NaN) },
        { type: 'depth_to_space', scale: 2 },
      ),
      /finite/,
    ],
    [
      'a missing bias',
      model({ ...conv(1, 4, false), bias: [] }, { type: 'depth_to_space', scale: 2 }),
      /bias/,
    ],
    [
      'an unknown layer type',
      model({ type: 'conv5x5' }, { type: 'depth_to_space', scale: 2 }),
      /conv3x3/,
    ],
    [
      'an absurd channel count',
      model(conv(1, 4, false), { ...conv(4, 4, false), out: 500 }, { type: 'depth_to_space' }),
      /in and out/,
    ],
  ];

  for (const [name, json, message] of broken) {
    it(`rejects ${name}`, () => {
      expect(() => parseModel(json)).toThrow(ModelError);
      expect(() => parseModel(json)).toThrow(message);
    });
  }
});

describe('weight packing', () => {
  it('pads channels to groups of four', () => {
    expect(groups(1)).toBe(1);
    expect(groups(4)).toBe(1);
    expect(groups(5)).toBe(2);
    expect(groups(16)).toBe(4);
    const layer = conv(1, 6, true);
    expect(packConvWeights(layer)).toHaveLength(2 * 1 * 9 * 16);
    expect(packBias(layer)).toHaveLength(8);
    expect(Array.from(packBias(layer).slice(6))).toEqual([0, 0]);
  });

  it('puts a single weight where the shader reads it', () => {
    // Output channel 5 (group 1, row 1), input channel 6 (group 1, column 2), tap ky=2, kx=0.
    const layer = conv(8, 8, false, () => 0);
    layer.weights[((5 * 8 + 6) * 3 + 2) * 3 + 0] = 1;
    const packed = packConvWeights(layer);
    const tap = 2 * 3 + 0;
    const matrix = (1 * 2 + 1) * 9 + tap;
    const index = matrix * 16 + 2 * 4 + 1;
    expect(packed[index]).toBe(1);
    expect(Array.from(packed).filter((v) => v !== 0)).toHaveLength(1);
  });

  // The packed layout drives the WGSL convolution; the PyTorch layout drives the reference. If
  // both agree for odd channel counts, padding and borders included, the packing is right.
  const shapes: [number, number, boolean][] = [
    [1, 4, true],
    [1, 8, true],
    [8, 8, true],
    [6, 5, false],
    [16, 4, false],
    [3, 9, true],
  ];
  for (const [inputs, outputs, relu] of shapes) {
    it(`matches the reference convolution for ${inputs} -> ${outputs}${relu ? ' with relu' : ''}`, () => {
      const layer = conv(inputs, outputs, relu);
      const width = 7;
      const height = 5;
      const input = planes(inputs, width, height);
      const expected = convReference(layer, input, width, height);
      const actual = convPackedReference(
        packConvWeights(layer),
        packBias(layer),
        groups(inputs),
        groups(outputs),
        relu,
        input,
        width,
        height,
      );
      for (let o = 0; o < outputs; o++) {
        for (let i = 0; i < width * height; i++) {
          expect(actual[o][i]).toBeCloseTo(expected[o][i], 4);
        }
      }
      // Padding channels stay exactly zero (bias 0, weights 0) and are never read downstream.
      for (let o = outputs; o < groups(outputs) * 4; o++) {
        expect(actual[o].every((v) => v === 0)).toBe(true);
      }
    });
  }
});

describe('CPU reference', () => {
  it('pads the convolution by clamping to the edge, like the training (replicate)', () => {
    // A 3x3 box filter over ones stays 9 everywhere: the edge repeats instead of reading zeros.
    const layer: Conv3x3Layer = {
      type: 'conv3x3',
      in: 1,
      out: 1,
      relu: false,
      weights: Array(9).fill(1),
      bias: [0],
    };
    const [out] = convReference(layer, [new Float32Array(9).fill(1)], 3, 3);
    expect(Array.from(out)).toEqual([9, 9, 9, 9, 9, 9, 9, 9, 9]);
  });

  it('takes the kernel the right way round (cross-correlation)', () => {
    // A kernel that only looks right: the output at x is the input at x+1 (the last repeats).
    const layer: Conv3x3Layer = {
      type: 'conv3x3',
      in: 1,
      out: 1,
      relu: false,
      weights: [0, 0, 0, 0, 0, 1, 0, 0, 0],
      bias: [0],
    };
    const [out] = convReference(layer, [Float32Array.from([1, 2, 3])], 3, 1);
    expect(Array.from(out)).toEqual([2, 3, 3]);
  });

  it('shuffles depth to space in PixelShuffle order', () => {
    // One source pixel, four channels: top-left, top-right, bottom-left, bottom-right.
    const input = [[1], [2], [3], [4]].map((v) => Float32Array.from(v));
    expect(Array.from(depthToSpaceReference(input, 2, 1, 1))).toEqual([1, 2, 3, 4]);
    const wide = [
      [1, 5],
      [2, 6],
      [3, 7],
      [4, 8],
    ].map((v) => Float32Array.from(v));
    expect(Array.from(depthToSpaceReference(wide, 2, 2, 1))).toEqual([1, 2, 5, 6, 3, 4, 7, 8]);
  });

  it('runs a whole network to a residual at twice the size', () => {
    const parsed = parseLinearModel(valid());
    const residual = runModelReference(parsed, planes(1, 6, 4)[0], 6, 4);
    expect(residual).toHaveLength(12 * 8);
    expect(residual.every(Number.isFinite)).toBe(true);
  });

  it('predicts a zero residual for an all-zero network', () => {
    const zero = (inputs: number, outputs: number) => ({
      ...conv(inputs, outputs, true),
      weights: Array(outputs * inputs * 9).fill(0),
      bias: Array(outputs).fill(0),
    });
    const parsed = parseLinearModel(
      model(zero(1, 4), zero(4, 4), { type: 'depth_to_space', scale: 2 }),
    );
    expect(runModelReference(parsed, planes(1, 3, 3)[0], 3, 3).every((v) => v === 0)).toBe(true);
  });
});

describe('half floats', () => {
  it('encodes exact values', () => {
    expect(toHalfBits(0)).toBe(0x0000);
    expect(toHalfBits(-0)).toBe(0x8000);
    expect(toHalfBits(1)).toBe(0x3c00);
    expect(toHalfBits(-2)).toBe(0xc000);
    expect(toHalfBits(0.5)).toBe(0x3800);
    expect(toHalfBits(65504)).toBe(0x7bff);
  });

  it('rounds to nearest, ties to even', () => {
    expect(toHalfBits(0.1)).toBe(0x2e66);
    expect(toHalfBits(1 + 2 ** -11)).toBe(0x3c00); // tie, even stays
    expect(toHalfBits(1 + 3 * 2 ** -11)).toBe(0x3c02); // tie, odd rounds up
  });

  it('handles subnormals, overflow and NaN', () => {
    expect(toHalfBits(2 ** -24)).toBe(0x0001);
    expect(toHalfBits(2 ** -15)).toBe(0x0200);
    expect(toHalfBits(1e-9)).toBe(0x0000);
    expect(toHalfBits(70000)).toBe(0x7c00);
    expect(toHalfBits(-Infinity)).toBe(0xfc00);
    expect(Number.isNaN(fromHalfBits(toHalfBits(NaN)))).toBe(true);
  });

  it('round-trips typical weights within half precision', () => {
    const random = rng(3);
    const values = Float32Array.from({ length: 500 }, () => random() * 2);
    const halves = toHalfArray(values);
    for (let i = 0; i < values.length; i++) {
      const back = fromHalfBits(halves[i]);
      expect(Math.abs(back - values[i])).toBeLessThanOrEqual(Math.abs(values[i]) * 2 ** -11 + 1e-7);
    }
  });
});

describe('loadModel', () => {
  const response = (ok: boolean, body: unknown, status = 200) =>
    Promise.resolve({ ok, status, json: () => Promise.resolve(body) });

  it('takes an object, a URL or a function', async () => {
    expect((await loadModel(valid())).scale).toBe(2);
    expect((await loadModel(async () => valid())).scale).toBe(2);
    let fetched = '';
    const fetchJson = (url: string) => {
      fetched = url;
      return response(true, valid());
    };
    expect((await loadModel('/weights/la-s.json', fetchJson)).scale).toBe(2);
    expect(fetched).toBe('/weights/la-s.json');
  });

  it('rejects an HTTP error and an invalid file', async () => {
    await expect(loadModel('/missing.json', () => response(false, null, 404))).rejects.toThrow(
      /404/,
    );
    await expect(loadModel({ scale: 2, layers: [] })).rejects.toThrow(ModelError);
  });
});
