import { describe, expect, it } from 'vitest';
import { isHdrTransfer } from '../src/gpu';

describe('HDR detection', () => {
  it('treats PQ and HLG as HDR, everything else as SDR', () => {
    expect(isHdrTransfer('pq')).toBe(true);
    expect(isHdrTransfer('hlg')).toBe(true);
    for (const t of ['bt709', 'srgb', 'smpte170m', 'linear', '', null, undefined]) {
      expect(isHdrTransfer(t)).toBe(false);
    }
  });
});
