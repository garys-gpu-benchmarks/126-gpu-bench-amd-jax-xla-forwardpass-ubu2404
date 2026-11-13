# Results: 126 JAX XLA Transformer Forward Pass

**Main goal:** Measure JAX/XLA transformer forward-pass performance  
**Tools:** Run gpu-bench-jax-xla-forwardpass.py via collect_jax_xla.py

Reference results measured by the maintainer on the system shown in each section (generator TEMPLATE_00_138 · recorded 2026-11-12). Your own runs write `results/RESULTS_ACTUAL.md` in the same format.

> Smoke runs use deliberately small sizes. They check that the workload installs and runs end to end; their numbers are not performance results. Baseline and extended are.

<!-- profile:smoke -->
## smoke

- **System:** 1× AMD Instinct MI300X VF (196288 MB) · INTEL(R) XEON(R) PLATINUM 8568Y+ · Ubuntu 24.04.5 LTS · kernel 6.8.0-147-generic · driver 7.1.9.31600000
- **Run:** 2026-11-12T23:32:52Z · status **PASS** (exit 0)
- **Time (mm:ss):** setup 01:12 · benchmark 00:25 · whole run 01:53
- **Source:** run 126_20261112_233252
- **Command:** `bash run_benchmark.sh --smoke --device-id 0 --dtype FP16 --sequence-len 16 --hidden-size 64 --num-layers 1 --num-heads 4 --vocab-size 32000 --batch-size 1 --warmup-iters 1 --num-iterations 3 --seed 42`

| # | Metric | Result |
|---|---|---|
| 1 | Forward-pass latency, ms (forward_latency_msec) | 0.10270666666666667 |
| 2 | Throughput, tokens/s (tokens_per_sec) | 155796.73162366665 |
| 3 | Modeled TFLOPS, full forward pass (tflops) | 0.6489913333333334 |
| 4 | XLA compilation time, ms (jit_compile_msec) | 3021.7603846666666 |
| 5 | Peak HBM, GB (peak_hbm_gb) | 0.188677 |
<!-- /profile:smoke -->
