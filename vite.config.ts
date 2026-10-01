import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/',
  server: { port: Number(process.env.PORT) || 5173 },
  build: {
    rollupOptions: {
      input: {
        main: 'index.html',
        klNllCe: 'posts/kl-nll-ce/index.html',
        rmsnorm: 'posts/rmsnorm/index.html',
        cudaGraph: 'posts/cuda-graph/index.html',
        flashAttention: 'posts/flash-attention/index.html',
        gradientCheckpointing: 'posts/gradient-checkpointing/index.html',
        compressionHarness: 'posts/compression-harness/index.html',
        rtxPro6000Topology: 'posts/rtx-pro-6000-topology/index.html',
        vllmPagedAttention: 'posts/vllm-paged-attention/index.html',
        vllmScheduler: 'posts/vllm-scheduler/index.html',
        vllmPrefixCaching: 'posts/vllm-prefix-caching/index.html',
        vllmSpeculativeDecoding: 'posts/vllm-speculative-decoding/index.html',
        vllmCpuOverhead: 'posts/vllm-cpu-overhead/index.html',
      },
    },
  },
})
