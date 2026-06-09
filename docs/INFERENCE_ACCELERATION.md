# Adam Inference Acceleration Plan

Branch: `dev/inference-accel` (off `main`). Goal: raise the **closed-loop frequency** (how
often the model recalculates a new plan from fresh camera feedback) for real-robot serving of the
Adam Stage-A full-FT checkpoint (`serve_wam.py` + `deploy_adam.py`).

## Where we are (measured)
- Per-inference latency (2-GPU TP on 2× RTX PRO 6000 Blackwell, checkpoint-7500): **~3.08 s ⇒ 0.32 Hz**.
- Profile (server `Time taken:` line, video-save off):
  - **Diffusion (DiT denoising): 2.45 s (~80%)** ← the bottleneck
  - Image encoder (CLIP): 0.39 s · KV-cache creation: 0.18 s · Text enc 0.03 s · VAE 0.00 · Scheduler 0.00
  - `WAM_NUM_INFERENCE_STEPS=16` ⇒ **8 actual DiT computes** (`--enable-dit-cache` skips ~half).
- Already applied: **CFG parallelism** (2-GPU TP), **DiT caching** (`--enable-dit-cache`),
  **async prefetch**, **world-model video saving disabled** (`serve_wam.py save_video_dir=""`, saved ~0.46 s).

## Target / reality
DreamZero paper (arXiv:2602.15922) hits **7 Hz / 150 ms via a 38× stack on GB200**. We are on
RTX PRO 6000 (workstation Blackwell, slower per-GPU, no GB200 interconnect), so **7 Hz is not
reachable here**; realistic target **~1–3 Hz**. The paper's stack, mapped to us:

| Layer | Paper cumulative | Us |
|---|---|---|
| CFG parallelism | 1.8× | ✅ have |
| DiT caching | 5.4× | ✅ have (16→8 computes) |
| Torch Compile + CUDA Graphs | 10.9× | ❌ missing (DiT `_forward_blocks` not compiled) |
| Kernel & scheduler opts | 14.8× | ❌ NVIDIA-internal, hard to replicate |
| Quantization (NVFP4) | 16.6× | ❌ feasible (Blackwell = FP4-capable), needs tooling |
| DreamZero-Flash (1-step) | 38× | ❌ **EXCLUDED** — needs retraining (decoupled video/action noise, `Beta(7,1)`) |

Paper deploy facts: **4 steps = 83% task progress; 1 step (no Flash) = 52%; Flash 1 step = 74%.**
So **4 steps is the no-retrain sweet spot**; 1-step requires Flash training.

## Plan (ranked; non-Flash only)

1. **Reduce denoising steps 16 → 4** — `WAM_NUM_INFERENCE_STEPS=4` (env, read at server start).
   **DONE / measured (2026-06-09):** total **3.08 s → 1.52 s (0.32 → 0.66 Hz, ~2×)**; diffusion
   2.45 → 0.91 s (8 → 3 DiT computes). Quality: paper's 4-step baseline = 83% — **still need to
   eyeball a real fold** before committing the default. After this, fixed overhead (image-encoder
   0.39 s + KV 0.18 s = 0.57 s) is now ~37% of latency → next target.
2. **Tune DiT cache more aggressively** — skip more steps if quality holds (it already does 16→8).
3. **torch.compile + CUDA Graphs on the DiT blocks** — the biggest no-quality-loss lever (~2×). The repo
   deliberately left `Wan _forward_blocks` uncompiled; blocker is dynamic shapes from the causal-chunk
   KV-cache → needs static-shape work / graph-break elimination. Medium-high effort.
4. **TensorRT engine** — `scripts/inference/build_trt_engine.py` exists; bundles compile + kernels + quant.
   May need updating for the causal-chunk attention. Medium-high effort, biggest single win.
5. **NVFP4 quantization** — Blackwell-capable; needs NVIDIA TensorRT-Model-Optimizer + calibration +
   integration; re-validate quality. High effort, ~1.1×.
6. **KV-cache streaming** — persist the causal KV-cache across chunks instead of rebuilding (~0.18 s),
   replacing generated frames with GT observations after each execution (paper's approach).
7. **(Excluded) DreamZero-Flash** — retrain with decoupled video/action noise schedule to enable
   1-step inference at good quality. Off the table unless we retrain.

## Order of attack
(1) steps→4 + cache tuning [today] → (3/4) torch.compile/TensorRT the DiT [the real project] →
(5) NVFP4 / (6) KV streaming if more is needed.

## Notes
- `WAM_NUM_INFERENCE_STEPS` = denoising steps to GENERATE one plan (server/model) — NOT the open-loop
  horizon (`--open-loop-horizon`, client execution) nor `--inference-freq` (playback rate, default 30).
- Video LATENT generation is load-bearing for actions (joint world-action DiT; action tokens attend to
  video tokens) — cannot skip. Video DECODE (latents→pixels) is not needed and is already disabled.
