import { defineConfig } from 'vitest/config';

// `tsc` emits the ESM build and the types (dist/index.js + .d.ts). Vite only produces the
// single-file browser global, `artplayerPluginLiveActionSR`, for a plain <script> tag - the same
// shape as the official ArtPlayer plugins. Everything, the WGSL shaders included, is inlined into it.
export default defineConfig({
  build: {
    outDir: 'dist',
    emptyOutDir: false,
    sourcemap: true,
    lib: {
      entry: 'src/global.ts',
      name: 'artplayerPluginLiveActionSR',
      formats: ['iife'],
      fileName: () => 'artplayer-plugin-liveaction-sr.js',
    },
    rolldownOptions: {
      output: {
        exports: 'default',
      },
    },
  },
  test: {
    include: ['test/**/*.test.ts'],
    environment: 'node',
  },
});
