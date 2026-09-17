import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  base: './',
  publicDir: 'data',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'esnext',
  },
});
