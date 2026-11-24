import type { LiveActionSrOptions } from '../src/index';
import { testModel } from './test-model';

/**
 * The demo pages' `models` option, from the query string:
 *
 * - `?model=shipped` loads the weights in `models/`: balanced, quality and ultra (EfRLFN);
 * - `?model=test` plugs random stand-in weights into balanced and quality, to run and time the CNN
 *   path (the picture stays near the bilinear base);
 * - `?model=<url>` loads that file for balanced and quality;
 * - `?ultra=<url>` (or `?ultra=shipped`) sets ultra on its own, alongside any of the above.
 */
export function demoModels(params: URLSearchParams): LiveActionSrOptions['models'] {
  const model = params.get('model');
  const ultra = params.get('ultra');
  const models: NonNullable<LiveActionSrOptions['models']> = {};
  if (model === 'shipped') {
    models.balanced = '/models/balanced.json';
    models.quality = '/models/quality.json';
    models.ultra = '/models/ultra.json';
  } else if (model === 'test') {
    models.balanced = testModel(8, 2);
    models.quality = testModel(16, 4);
  } else if (model) {
    models.balanced = model;
    models.quality = model;
  }
  if (ultra) models.ultra = ultra === 'shipped' ? '/models/ultra.json' : ultra;
  return Object.keys(models).length > 0 ? models : undefined;
}
