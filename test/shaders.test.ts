import { describe, expect, it } from 'vitest';
import notices from '../THIRD_PARTY_NOTICES.md?raw';
import fsrSource from '../src/shaders/fsr1.wgsl.ts?raw';
import { DEBAND_WGSL } from '../src/shaders/deband.wgsl';
import { EASU_WGSL, RCAS_WGSL } from '../src/shaders/fsr1.wgsl';
import { convWgsl, lumaWgsl, reconstructWgsl } from '../src/shaders/model.wgsl';
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
  ['luma', lumaWgsl('bt709')],
  ['conv', convWgsl(2, 4, true, false)],
  ['conv f16', convWgsl(4, 1, false, true)],
  ['reconstruct', reconstructWgsl(2)],
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

  it('present has a vertex and a fragment entry point', () => {
    expect(balanced(PRESENT_WGSL)).toBe(true);
    expect(PRESENT_WGSL).toContain('fn vert_main');
    expect(PRESENT_WGSL).toContain('fn frag_main');
  });

  it('specializes the convolution', () => {
    const code = convWgsl(3, 2, true, false);
    expect(code).toContain('const INPUT_GROUPS : u32 = 3u;');
    expect(code).toContain('const OUTPUT_GROUPS : u32 = 2u;');
    expect(code).toContain('max(acc');
    expect(code).not.toContain('enable f16');
    expect(convWgsl(1, 1, false, false)).not.toContain('max(acc');
  });

  it('enables f16 only on request, as the first statement', () => {
    const code = convWgsl(1, 1, false, true);
    expect(code.trimStart().startsWith('enable f16;')).toBe(true);
    expect(code).toContain('array<mat4x4<f16>>');
  });

  it('uses the luma coefficients the network was trained with', () => {
    expect(lumaWgsl('bt709')).toContain('0.2126, 0.7152, 0.0722');
    expect(lumaWgsl('bt601')).toContain('0.299, 0.587, 0.114');
  });

  it("keeps AMD's copyright notice with the FSR port and in the package notices", () => {
    const notice = 'Copyright (c) 2021 Advanced Micro Devices, Inc.';
    expect(fsrSource).toContain(notice);
    expect(notices).toContain(notice);
  });
});
