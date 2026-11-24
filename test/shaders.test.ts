import { describe, expect, it } from 'vitest';
import notices from '../THIRD_PARTY_NOTICES.md?raw';
import fsrSource from '../src/shaders/fsr1.wgsl.ts?raw';
import { DEBAND_WGSL } from '../src/shaders/deband.wgsl';
import { EASU_WGSL, RCAS_WGSL } from '../src/shaders/fsr1.wgsl';
import {
  GRAPH_POOL_FINAL_WGSL,
  GRAPH_POOL_PARTIAL_WGSL,
  graphChannelConvWgsl,
  graphConvWgsl,
  graphElementwiseWgsl,
  graphInputWgsl,
  graphRgbOutputWgsl,
  reconstructWgsl,
  type ConvShader,
} from '../src/shaders/graph.wgsl';
import { PRESENT_WGSL } from '../src/shaders/present.wgsl';

// The shaders only compile in a browser (see the demo). These checks catch what a template
// mistake would break before that: unbalanced code, missing entry points, wrong specialization.

function balanced(code: string): boolean {
  const pairs: Record<string, string> = { ')': '(', ']': '[', '}': '{' };
  const stack: string[] = [];
  for (const char of code.replace(/\/\/.*$/gm, '')) {
    if ('([{'.includes(char)) stack.push(char);
    else if (char in pairs && stack.pop() !== pairs[char]) return false;
  }
  return stack.length === 0;
}

const compute: [string, string][] = [
  ['deband', DEBAND_WGSL],
  ['easu', EASU_WGSL],
  ['rcas', RCAS_WGSL],
  ['reconstruct', reconstructWgsl(2)],
];

const conv: ConvShader = {
  inputGroups: 13,
  outputGroups: 14,
  storedGroups: 13,
  perInvocation: 2,
  pixels: 8,
  kernel: 3,
  dilation: 1,
  padding: 'zero',
  activation: 'tanh',
  slope: 0.01,
  f16: true,
  accumulateF32: true,
  format: 'rgba16float',
};

const graph: [string, string][] = [
  ['graph input rgb', graphInputWgsl('rgb', 'bt709', 'rgba16float')],
  ['graph input luma', graphInputWgsl('luma', 'bt601', 'rgba32float')],
  ['graph conv', graphConvWgsl(conv)],
  ['graph conv replicate f32', graphConvWgsl({ ...conv, padding: 'replicate', f16: false })],
  ['graph conv 5x5 dilated', graphConvWgsl({ ...conv, kernel: 5, dilation: 2, pixels: 4 })],
  ['graph add', graphElementwiseWgsl('add', false, 'none', 0, 'rgba16float')],
  ['graph mul vector', graphElementwiseWgsl('mul', true, 'lrelu', 0.2, 'rgba16float')],
  ['graph activation', graphElementwiseWgsl(null, false, 'sigmoid', 0, 'rgba32float')],
  ['graph pool partial', GRAPH_POOL_PARTIAL_WGSL],
  ['graph rgb out', graphRgbOutputWgsl(2)],
];

describe('WGSL sources', () => {
  for (const [name, code] of compute) {
    it(`${name} is balanced and has a compute entry point`, () => {
      expect(balanced(code)).toBe(true);
      expect(code).toMatch(/@compute @workgroup_size\(8, 8, 1\)\s*fn main/);
      expect(code).not.toContain('${');
      expect(code).not.toContain('undefined');
    });
  }

  for (const [name, code] of [
    ...graph,
    ['graph pool final', GRAPH_POOL_FINAL_WGSL],
    ['graph channel conv', graphChannelConvWgsl(52, 3, 'sigmoid', 0)],
  ]) {
    it(`${name} is balanced and has a compute entry point`, () => {
      expect(balanced(code)).toBe(true);
      expect(code).toMatch(/@compute @workgroup_size\(\d+, \d+, \d+\)\s*fn main/);
      expect(code).not.toContain('${');
      expect(code).not.toContain('undefined');
      expect(code).not.toContain('NaN');
    });
  }

  it('specializes the graph convolution', () => {
    const code = graphConvWgsl(conv);
    expect(code.trimStart().startsWith('enable f16;')).toBe(true);
    expect(code).toContain('const STORED_GROUPS : u32 = 13u;');
    // 2 groups x 8 pixels of accumulators, in f32; 10 texels per row for a 3x3 over 8 pixels.
    expect(code.match(/var a\d+_\d+ = vec4<f32>/g)).toHaveLength(16);
    expect(code.match(/let t\d+ = /g)).toHaveLength(10);
    expect(code).toContain('select(vec4f(0.0)');
    expect(code).toContain('tanh(');
    const replicate = graphConvWgsl({ ...conv, padding: 'replicate', f16: false });
    expect(replicate).not.toContain('select(vec4f(0.0)');
    expect(replicate).not.toContain('enable f16');
    expect(graphConvWgsl({ ...conv, accumulateF32: false })).toContain('var a0_0 = vec4<f16>');
    expect(graphConvWgsl({ ...conv, format: 'rgba32float' })).toContain('<rgba32float, write>');
  });

  it('present has a vertex and a fragment entry point', () => {
    expect(balanced(PRESENT_WGSL)).toBe(true);
    expect(PRESENT_WGSL).toContain('fn vert_main');
    expect(PRESENT_WGSL).toContain('fn frag_main');
  });

  it('uses the luma coefficients the network was trained with', () => {
    expect(graphInputWgsl('luma', 'bt709', 'rgba16float')).toContain('0.2126, 0.7152, 0.0722');
    expect(graphInputWgsl('luma', 'bt601', 'rgba16float')).toContain('0.299, 0.587, 0.114');
  });

  it("keeps AMD's copyright notice with the FSR port and in the package notices", () => {
    const notice = 'Copyright (c) 2021 Advanced Micro Devices, Inc.';
    expect(fsrSource).toContain(notice);
    expect(notices).toContain(notice);
  });
});
