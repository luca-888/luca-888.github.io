# Handwritten RMSNorm Backward: isolated shape experiment

RTX 4090, contiguous M=4096, N=8192; BF16 inputs/outputs, FP32 r and intermediates. Forward excluded.

- `kernel.py`: selected 32-row/8-warp fused implementation and allocation wrapper. It writes dx and FP32 dgamma partials together, then reduces the 128 partial rows in a second kernel. Workspace: 4 MiB. This is a tested shape specialization, not an arbitrary-shape or autograd implementation.
- `benchmark.py`: reference extraction, original saved compile kernels, handwritten candidate sweep, validation, and measurements. It reads the main repository without editing it.
- `results.json`: all candidate data, the final paired recheck, full-call timing, peak allocated memory and numerical errors. Initial selected GPU sweep: 222.97 us; final paired recheck: 208.82 us. Both are retained; final call timing independently measured 210.20 us. These are measurements for this run, not a guaranteed bound.

GPU timing compares identical preallocated inputs/outputs through CUDA graph replay: Triton do_bench_cudagraph, 30 ms, 10 samples per round, 3 rotated rounds, median of round medians, no cache flush. Compile kernels use the current capture stream. Pipeline timings include both kernels. Full-call timing separately uses torch.utils.benchmark.Timer, 30 warmups, blocked_autorange(.3 s), 3 rotated rounds; compile uses the actual torch.compile callable. First-call compilation excluded. Peak memory includes newly allocated outputs and workspace, excludes existing inputs and saved r; measured outside CUDA graphs. No allocation is hidden from the full-call memory result.

Numerics compare against the article reference with BF16 rtol=.016, atol=.001, random normal input plus x scaled by 0, .001 and 100. Absolute and relative L2 errors are retained. No claim of bitwise equality. This run does not establish behavior or speed for other shapes.

## Reproduce

From the repository root, using Python 3.12.14, torch 2.9.1+cu128 and Triton 3.5.1:

```bash
PYTHONDONTWRITEBYTECODE=1 TRITON_CACHE_DIR=/tmp/rmsnorm-fused-triton-cache TORCHINDUCTOR_CACHE_DIR=/tmp/rmsnorm-fused-inductor-cache python public/measurements/rmsnorm-fused/benchmark.py
```

The script reads the first two Python blocks in content/rmsnorm.md and the original compile backward.py; it writes measurements beside itself. The selected kernel.py is the implementation reported in the article. The script explores candidates, so the selected configuration can change on a different system. Full environment details are in environment.json. The extra seed-42 standalone validation in results.json was performed separately after exporting kernel.py.

The article table uses selected_gpu_recheck for GPU time, median(call_times) for full-call time, and memory.*.peak_increment_bytes / 2**20 for MiB. Initial sweep and final recheck are both preserved; GPU timings should not be compared across those phases as a single-variable ablation.
