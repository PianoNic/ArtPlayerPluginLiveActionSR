import type { LinearModel } from '../src/index';

/**
 * A stand-in network for the demo, until trained weights exist: the real layer shapes with tiny
 * random weights, so the CNN presets run (and their cost can be measured) while the residual stays
 * close to zero and the picture is essentially the bilinear base. `?model=test` turns it on.
 */
export function testModel(channels: number, hiddenLayers: number, seed = 1): LinearModel {
  let state = seed >>> 0 || 1;
  const random = () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return ((state >>> 0) / 4294967296 - 0.5) * 0.02;
  };
  const conv = (inputs: number, outputs: number, relu: boolean) => ({
    type: 'conv3x3' as const,
    in: inputs,
    out: outputs,
    relu,
    weights: Array.from({ length: outputs * inputs * 9 }, random),
    bias: Array.from({ length: outputs }, () => 0),
  });
  const layers: LinearModel['layers'] = [conv(1, channels, true)];
  for (let i = 0; i < hiddenLayers; i++) layers.push(conv(channels, channels, true));
  layers.push(conv(channels, 4, false), { type: 'depth_to_space', scale: 2 });
  return { arch: 'demo-random', scale: 2, channels, luma: 'bt709', layers };
}
