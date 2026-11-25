# Third-party notices

This package's own code is MIT-licensed (see [LICENSE](LICENSE)). It contains a port of the
following third-party code, which keeps its own copyright and licence.

## AMD FidelityFX Super Resolution 1.0 (FSR 1)

`src/shaders/fsr1.wgsl.ts` (and the bundled `dist/` files built from it) contains a WGSL port of
EASU and RCAS from `ffx_fsr1.h` of
[AMD FidelityFX Super Resolution](https://github.com/GPUOpen-Effects/FidelityFX-FSR).

```
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## Lucide

The default settings icon is Lucide's `film` glyph, [lucide.dev](https://lucide.dev), ISC licence:

```
Copyright (c) for portions of Lucide are held by Cole Bemis 2013-2022 as part of Feather (MIT).
All other copyright (c) for Lucide are held by Lucide Contributors 2022.

Permission to use, copy, modify, and/or distribute this software for any purpose with or without
fee is hereby granted, provided that the above copyright notice and this permission notice appear
in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE
AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
THIS SOFTWARE.
```

## EfRLFN (the `ultra` weights)

`models/ultra.json` holds the 2x weights of
[EfRLFN](https://github.com/EvgeneyBogatyrev/EfRLFN) (Bogatyrev et al., "Exploring Real-Time
Super-Resolution: Benchmarking and Fine-Tuning for Streaming Content", ICLR 2026,
arXiv:2602.11339), converted to this package's graph format by `tools/efrlfn/convert.py` in LiveActionSR (stored
as float16; the network itself is unchanged). No EfRLFN source code is included: the plugin runs the
network with its own WGSL shaders.

```
MIT License

Copyright (c) 2026 MSU Graphics & Media Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Not included

The deband, film-grain, CNN and present shaders were written for this project. No code from
libplacebo, mpv, f3kdb, RAVU, FSRCNNX, Anime4K, Magpie or EfRLFN is included. The weights in
`models/` are loaded at run time from wherever the host page points `models` at: `balanced.json`
and `quality.json` are this project's (MIT), `ultra.json` is EfRLFN's (MIT, above).
