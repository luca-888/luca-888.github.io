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
        llmQuantization: 'posts/llm-quantization/index.html',
        jev: 'posts/jev/index.html',
        vllmDistributed: 'posts/vllm-distributed/index.html',
        dspark: 'posts/dspark/index.html',
        attentionKvCompression: 'posts/attention-kv-compression/index.html',
        sparseAttention: 'posts/sparse-attention/index.html',
        linearAttention: 'posts/linear-attention/index.html',
        bf16WeightOffset: 'posts/bf16-weight-offset/index.html',
        ligerHfPatching: 'posts/liger-hf-patching/index.html',
        semanticCacheVerification: 'posts/semantic-cache-verification/index.html',
      },
    },
  },
})
