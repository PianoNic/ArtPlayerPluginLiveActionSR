import { describe, expect, it } from 'vitest';
import fixtureText from './fixtures/efrlfn-tiny.json?raw';
import ultraText from '../models/ultra.json?raw';
import modelNotice from '../models/NOTICE.md?raw';
import notices from '../THIRD_PARTY_NOTICES.md?raw';
import {
  graphConvPackedReference,
  graphConvReference,
  graphInputReference,
  isGraphModel,
  macsPerPixel,
  packConvBias,
  packConvKernel,
  parameterCount,
  parseGraphModel,
  runGraphReference,
  tensorShapes,
  type ConvNode,
  type GraphModel,
} from '../src/graph';
import {
  groups,
  ModelError,
  parseLinearModel,
  parseModel,
  runModelReference,
  toHalfBits,
} from '../src/model';
import { decodeTensor } from '../src/tensor';

/** Deterministic pseudo-random numbers in -1..1. */
function rng(seed: number) {
  let state = seed;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 2 ** 31 - 1;
  };
}

function planes(channels: number, width: number, height: number, seed = 7) {
  const random = rng(seed);
  return Array.from({ length: channels }, () =>
    Float32Array.from({ length: width * height }, random),
  );
}

function convJson(
  name: string,
  input: string,
  cin: number,
  cout: number,
  extra: Record<string, unknown> = {},
  random = rng(cin * 31 + cout + name.length),
) {
  const kernel = (extra.kernel as number | undefined) ?? 3;
  return {
    name,
    op: 'conv',
    inputs: [input],
    in: cin,
    out: cout,
    kernel,
    weights: Array.from({ length: cout * cin * kernel * kernel }, random),
    bias: Array.from({ length: cout }, random),
    ...extra,
  };
}

/** A small network with every op: skips, attention, a dilated 5x5, an lrelu. */
function everyOp(output: 'rgb' | 'luma_residual' = 'rgb') {
  const planesOut = output === 'rgb' ? 3 : 1;
  return {
    arch: 'test',
    scale: 2,
    input: output === 'rgb' ? 'rgb' : 'luma',
    output,
    nodes: [
      convJson('head', 'input', planesOut, 6, { activation: 'relu' }),
      convJson('wide', 'head', 6, 6, { kernel: 5, dilation: 2, padding: 'replicate' }),
      { name: 'skip', op: 'add', inputs: ['wide', 'head'], activation: 'tanh' },
      convJson('point', 'skip', 6, 6, { kernel: 1, activation: 'lrelu', slope: 0.2 }),
      { name: 'pool', op: 'global_avg_pool', inputs: ['point'] },
      {
        name: 'attention',
        op: 'channel_conv',
        inputs: ['pool'],
        kernel: 3,
        weights: [0.5, -1, 0.75],
        bias: [0.1],
        activation: 'sigmoid',
      },
      { name: 'scaled', op: 'mul', inputs: ['point', 'attention'] },
      { name: 'squash', op: 'activation', inputs: ['scaled'], activation: 'tanh' },
      convJson('tail', 'squash', 6, planesOut * 4),
      { name: 'shuffle', op: 'depth_to_space', inputs: ['tail'], scale: 2 },
    ],
  };
}

const fixture = JSON.parse(fixtureText) as {
  width: number;
  height: number;
  input: number[];
  output: number[];
  model: unknown;
};

describe('parseGraphModel', () => {
  it('accepts a graph with every op and tells it apart from the linear format', () => {
    const model = parseModel(everyOp());
    expect(isGraphModel(model)).toBe(true);
    const graph = model as GraphModel;
    expect(graph.nodes.map((n) => n.op)).toEqual([
      'conv',
      'conv',
      'add',
      'conv',
      'global_avg_pool',
      'channel_conv',
      'mul',
      'activation',
      'conv',
      'depth_to_space',
    ]);
    const wide = graph.nodes[1] as ConvNode;
    expect([wide.kernel, wide.dilation, wide.padding]).toEqual([5, 2, 'replicate']);
    // PyTorch defaults: zero padding, no activation, lrelu slope 0.01.
    expect((graph.nodes[0] as ConvNode).padding).toBe('zero');
    expect((graph.nodes[8] as ConvNode).activation).toBe('none');
    expect(tensorShapes(graph).get('pool')).toEqual({ channels: 6, vector: true });
    expect(tensorShapes(graph).get('shuffle')).toEqual({ channels: 3, vector: false });
  });

  it('still reads linear files as linear', () => {
    const linear = {
      scale: 2,
      layers: [
        {
          type: 'conv3x3',
          in: 1,
          out: 4,
          relu: false,
          weights: Array(36).fill(0),
          bias: [0, 0, 0, 0],
        },
        { type: 'depth_to_space', scale: 2 },
      ],
    };
    expect(isGraphModel(parseModel(linear))).toBe(false);
  });

  const nodes = () => everyOp().nodes as Record<string, unknown>[];
  const withNodes = (list: unknown[]) => ({ ...everyOp(), nodes: list });
  const replace = (index: number, patch: Record<string, unknown>) => {
    const list = nodes();
    list[index] = { ...list[index], ...patch };
    return withNodes(list);
  };
  const broken: [string, unknown, RegExp][] = [
    ['a bad input', { ...everyOp(), input: 'yuv' }, /input must be/],
    ['a bad output', { ...everyOp(), output: 'luma' }, /output must be/],
    ['an unknown op', replace(2, { op: 'concat' }), /unknown op concat/],
    ['a forward reference', replace(1, { inputs: ['skip'] }), /not the frame or an earlier node/],
    ['a duplicate name', replace(1, { name: 'head' }), /already taken/],
    ['the name of the frame', replace(0, { name: 'input' }), /already taken/],
    ['a wrong in', replace(1, { in: 5 }), /in is 5, but head has 6/],
    ['an even kernel', replace(1, { kernel: 4 }), /kernel must be one of/],
    ['a bad padding', replace(1, { padding: 'reflect' }), /padding/],
    ['a bad dilation', replace(1, { dilation: 0 }), /dilation/],
    ['too few weights', replace(0, { weights: [1, 2] }), /weights must be 162 finite numbers/],
    ['an unknown activation', replace(0, { activation: 'gelu' }), /activation must be/],
    ['a vector into a conv', replace(8, { inputs: ['pool'] }), /must be a map/],
    ['a map into channel_conv', replace(5, { inputs: ['point'] }), /must be a vector/],
    ['an add of different widths', replace(2, { inputs: ['wide', 'input'] }), /differ in channels/],
    ['an add with one input', replace(2, { inputs: ['wide'] }), /inputs must name 2 tensors/],
    [
      'depth_to_space before the end',
      withNodes([...nodes().slice(0, 9), nodes()[9], { ...nodes()[9], name: 'again' }]),
      /must be the last node/,
    ],
    ['no depth_to_space', withNodes(nodes().slice(0, 9)), /must be the last node/],
    [
      'an output with the wrong channel count',
      withNodes([
        ...nodes().slice(0, 8),
        convJson('tail', 'squash', 6, 4),
        { name: 'shuffle', op: 'depth_to_space', inputs: ['tail'], scale: 2 },
      ]),
      /rgb output needs 12 channels, got 4/,
    ],
    ['a mismatched scale', replace(9, { scale: 3 }), /scale must match/],
  ];
  for (const [name, json, message] of broken) {
    it(`rejects ${name}`, () => {
      expect(() => parseGraphModel(json)).toThrow(ModelError);
      expect(() => parseGraphModel(json)).toThrow(message);
    });
  }
});

describe('weight tensors', () => {
  const bytes = (values: number[], half: boolean) => {
    const view = new DataView(new ArrayBuffer(values.length * (half ? 2 : 4)));
    values.forEach((v, i) =>
      half ? view.setUint16(i * 2, toHalfBits(v), true) : view.setFloat32(i * 4, v, true),
    );
    return btoa(String.fromCharCode(...new Uint8Array(view.buffer)));
  };

  it('decodes plain numbers, float16 and float32 base64', () => {
    expect(Array.from(decodeTensor([1, -2.5], 2, 'w'))).toEqual([1, -2.5]);
    const values = [0.5, -0.25, 3, 0.1];
    const half = decodeTensor({ dtype: 'float16', base64: bytes(values, true) }, 4, 'w');
    expect(Array.from(half)).toEqual([0.5, -0.25, 3, 0.0999755859375]);
    const full = decodeTensor({ dtype: 'float32', base64: bytes(values, false) }, 4, 'w');
    expect(Array.from(full)).toEqual(Array.from(Float32Array.from(values)));
  });

  it('rejects a wrong length, an unknown dtype, bad base64 and infinities', () => {
    expect(() => decodeTensor({ dtype: 'float16', base64: bytes([1, 2], true) }, 3, 'w')).toThrow(
      /3 float16 values, got 4 bytes/,
    );
    expect(() => decodeTensor({ dtype: 'bfloat16', base64: '' }, 0, 'w')).toThrow(/dtype/);
    expect(() => decodeTensor({ dtype: 'float32', base64: '***' }, 1, 'w')).toThrow(/base64/);
    expect(() =>
      decodeTensor({ dtype: 'float32', base64: bytes([Infinity], false) }, 1, 'w'),
    ).toThrow(/finite/);
    expect(() => decodeTensor('1,2', 2, 'w')).toThrow(/finite numbers/);
  });
});

describe('graph convolution packing', () => {
  const node = (cin: number, cout: number, extra: Record<string, unknown> = {}) => {
    const graph = parseGraphModel({
      scale: 2,
      input: 'rgb',
      output: 'rgb',
      nodes: [
        { name: 'pre', op: 'activation', inputs: ['input'] },
        ...(cin === 3 ? [] : [convJson('widen', 'pre', 3, cin)]),
        convJson('conv', cin === 3 ? 'pre' : 'widen', cin, cout, extra),
        convJson('tail', 'conv', cout, 12),
        { name: 'out', op: 'depth_to_space', inputs: ['tail'], scale: 2 },
      ],
    });
    return graph.nodes.find((n) => n.name === 'conv') as ConvNode;
  };

  it('puts a single weight where the shader reads it', () => {
    // Output channel 5 (group 1, row 1), input channel 6 (group 1, column 2), tap ky=2, kx=0.
    const conv = node(8, 8);
    conv.weights.fill(0);
    conv.weights[((5 * 8 + 6) * 3 + 2) * 3 + 0] = 1;
    const packed = packConvKernel(conv);
    const matrix = (1 * 2 + 1) * 9 + 2 * 3 + 0;
    expect(packed[matrix * 16 + 2 * 4 + 1]).toBe(1);
    expect(Array.from(packed).filter((v) => v !== 0)).toHaveLength(1);
    expect(packConvBias(node(6, 5))).toHaveLength(8);
  });

  const cases: [string, number, number, Record<string, unknown>, number][] = [
    ['3x3 zero padded', 6, 5, {}, 2],
    ['1x1', 8, 9, { kernel: 1, activation: 'tanh' }, 3],
    ['5x5 dilated, replicate', 3, 4, { kernel: 5, dilation: 2, padding: 'replicate' }, 1],
    ['7x7 sigmoid', 5, 7, { kernel: 7, activation: 'sigmoid' }, 4],
    ['3x3 dilated 3, zero, lrelu', 4, 13, { dilation: 3, activation: 'lrelu', slope: 0.3 }, 4],
  ];
  for (const [name, cin, cout, extra, per] of cases) {
    it(`matches the PyTorch-layout reference: ${name}, ${per} groups per invocation`, () => {
      const conv = node(cin, cout, extra);
      const padded = Math.ceil(groups(cout) / per) * per;
      const width = 9;
      const height = 6;
      const input = planes(cin, width, height);
      const expected = graphConvReference(conv, input, width, height);
      const packed = packConvKernel(conv, padded);
      expect(packed).toHaveLength(padded * groups(cin) * conv.kernel ** 2 * 16);
      const actual = graphConvPackedReference(
        conv,
        packed,
        packConvBias(conv, padded),
        input,
        width,
        height,
      );
      for (let o = 0; o < cout; o++) {
        for (let i = 0; i < width * height; i++) {
          expect(actual[o][i]).toBeCloseTo(expected[o][i], 5);
        }
      }
    });
  }

  it('pads with zeros, not with the edge, by default', () => {
    // A 3x3 box over ones: 4 taps in a corner, 6 on an edge, 9 inside.
    const conv = { ...node(3, 3), in: 1, out: 1, weights: new Float32Array(9).fill(1) };
    conv.bias = new Float32Array(1);
    const [out] = graphConvReference(conv, [new Float32Array(9).fill(1)], 3, 3);
    expect(Array.from(out)).toEqual([4, 6, 4, 6, 9, 6, 4, 6, 4]);
  });
});

describe('graph CPU reference', () => {
  it('matches PyTorch on a tiny random EfRLFN (fixture from tools/efrlfn)', () => {
    const model = parseGraphModel(fixture.model);
    expect(model.arch).toBe('efrlfn');
    const { width, height } = fixture;
    const area = width * height;
    const input = [0, 1, 2].map((c) =>
      Float32Array.from(fixture.input.slice(c * area, (c + 1) * area)),
    );
    const output = runGraphReference(model, input, width, height);
    expect(output).toHaveLength(3);
    let worst = 0;
    output.forEach((plane, c) => {
      plane.forEach((v, i) => {
        worst = Math.max(worst, Math.abs(v - fixture.output[c * area * 4 + i]));
      });
    });
    expect(worst).toBeLessThan(1e-4);
  });

  it('runs the linear format as a graph to the same residual', () => {
    const random = rng(11);
    const layer = (cin: number, cout: number, relu: boolean) => ({
      type: 'conv3x3' as const,
      in: cin,
      out: cout,
      relu,
      weights: Array.from({ length: cout * cin * 9 }, random),
      bias: Array.from({ length: cout }, random),
    });
    const layers = [layer(1, 6, true), layer(6, 6, true), layer(6, 4, false)];
    const linear = parseLinearModel({
      scale: 2,
      layers: [...layers, { type: 'depth_to_space', scale: 2 }],
    });
    const names = ['a', 'b', 'c'];
    const graph = parseGraphModel({
      scale: 2,
      input: 'luma',
      output: 'luma_residual',
      nodes: [
        ...layers.map((l, i) => ({
          name: names[i],
          op: 'conv',
          inputs: [i === 0 ? 'input' : names[i - 1]],
          in: l.in,
          out: l.out,
          kernel: 3,
          padding: 'replicate',
          activation: l.relu ? 'relu' : 'none',
          weights: l.weights,
          bias: l.bias,
        })),
        { name: 'shuffle', op: 'depth_to_space', inputs: ['c'], scale: 2 },
      ],
    });
    const rgb = planes(3, 6, 5, 5) as [Float32Array, Float32Array, Float32Array];
    const luma = graphInputReference(graph, rgb);
    const [residual] = runGraphReference(graph, luma, 6, 5);
    const expected = runModelReference(linear, luma[0], 6, 5);
    for (let i = 0; i < expected.length; i++) expect(residual[i]).toBeCloseTo(expected[i], 5);
  });

  it('pools, convolves across channels and scales each channel', () => {
    // The tail copies channel c to the top-left sub-pixel of output plane c.
    const tail: number[] = Array(12 * 3).fill(0);
    for (let c = 0; c < 3; c++) tail[c * 4 * 3 + c] = 1;
    const graph = parseGraphModel({
      scale: 2,
      input: 'rgb',
      output: 'rgb',
      nodes: [
        { name: 'pool', op: 'global_avg_pool', inputs: ['input'] },
        { name: 'att', op: 'channel_conv', inputs: ['pool'], kernel: 3, weights: [1, 10, 100] },
        { name: 'scaled', op: 'mul', inputs: ['input', 'att'] },
        { name: 'tail', op: 'conv', inputs: ['scaled'], in: 3, out: 12, kernel: 1, weights: tail },
        { name: 'out', op: 'depth_to_space', inputs: ['tail'], scale: 2 },
      ],
    });
    // Channel means 1, 2, 3; zero-padded taps (c-1, c, c+1) give 0+10+200, 1+20+300, 2+30+0.
    const input = [1, 2, 3].map((v) => Float32Array.from([v - 0.5, v + 0.5]));
    const out = runGraphReference(graph, input, 2, 1);
    const attention = [210, 321, 32];
    for (let c = 0; c < 3; c++) {
      expect(out[c][0]).toBeCloseTo((c + 0.5) * attention[c], 3);
      expect(out[c][2]).toBeCloseTo((c + 1.5) * attention[c], 3);
    }
  });
});

describe('models/ultra.json', () => {
  const ultra = parseModel(JSON.parse(ultraText));

  it('is the EfRLFN x2 graph, RGB in and out', () => {
    expect(isGraphModel(ultra)).toBe(true);
    const graph = ultra as GraphModel;
    expect(graph.arch).toBe('efrlfn');
    expect(graph.scale).toBe(2);
    expect([graph.input, graph.output]).toEqual(['rgb', 'rgb']);
    expect(parameterCount(graph)).toBe(487010);
    expect(macsPerPixel(graph)).toBe(485628);
    expect(graph.nodes.filter((n) => n.op === 'global_avg_pool')).toHaveLength(6);
  });

  it("ships with EfRLFN's MIT notice and citation", () => {
    const notice = 'Copyright (c) 2026 MSU Graphics & Media Lab';
    expect(modelNotice).toContain(notice);
    expect(modelNotice).toContain('arXiv:2602.11339');
    expect(notices).toContain(notice);
  });
});
