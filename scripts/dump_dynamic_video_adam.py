#!/usr/bin/env python3
"""Dump the predicted "dynamic" (world-model) video from an Adam Stage-A LoRA checkpoint.

Structural clone of scripts/open_loop_adam.py, but instead of discarding the
video the forward returns, we capture it, accumulate across steps, VAE-decode
to pixels, and write an MP4. This lets us SEE what the video (dynamics) head
predicts — useful because the dynamics_loss is flat.

What it produces in --output_dir:
  pred_dynamic.mp4   : decoded video_pred (the model's predicted 2x2 grid)
  gt_grid.mp4        : the ground-truth 2x2 grid (same layout the model sees),
                       for side-by-side comparison
  prompt.txt         : the exact prompt fed to the text encoder (decoded from
                       the token ids the transform produced) + raw task

Interpretation (per the handoff plan):
  coherent-but-blurry pred  -> pipeline is fine; flat dynamics_loss is a
                               training/measurement issue, not a feed bug
  pure noise / static pred  -> loading / conditioning bug

Usage (single GPU, avoid GPU 1):
  CUDA_VISIBLE_DEVICES=2 python scripts/dump_dynamic_video_adam.py \
      --model_path ./checkpoints/adam_stage_a_lora/checkpoint-10000 \
      --dataset_path ./data \
      --device cuda:0 --num_samples 48 --use_dataset_prompt \
      --output_dir results_dynamic_10000
"""

import torch._dynamo
torch._dynamo.config.disable = True

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from tianshou.data import Batch

# Reuse the tested dataset reader + obs builder from the open-loop eval script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from open_loop_adam import (  # noqa: E402
    AdamDataset,
    build_obs,
    VIDEO_CAMERAS,
)

from groot.vla.data.schema import EmbodimentTag  # noqa: E402
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy  # noqa: E402


# ---------------------------------------------------------------------------
# Prompt verification: spy on the collate() that builds the text the model sees
# ---------------------------------------------------------------------------

def install_prompt_spy(capture: dict):
    """Monkeypatch dreamzero_cotrain.collate to capture/decode the exact prompt.

    apply_batch() calls the module-global collate() by name, so replacing the
    module attribute intercepts the real call. We decode the produced token ids
    back to text so we see EXACTLY what the text encoder receives (including the
    YAM multi-view template wrapping for the adam embodiment, id 32)."""
    import groot.vla.model.dreamzero.transform.dreamzero_cotrain as dct

    orig = dct.collate

    def spy(features, tokenizer, num_views=3, embodiment_tag_mapping=None):
        out = orig(features, tokenizer, num_views, embodiment_tag_mapping)
        if not capture.get("done"):
            try:
                emb_ids = [f.get("embodiment_id") for f in features]
                ids = out["text"]
                decoded = tokenizer.tokenizer.batch_decode(ids, skip_special_tokens=True)
                capture["embodiment_id"] = emb_ids
                capture["decoded"] = decoded
            except Exception as e:  # pragma: no cover
                capture["error"] = repr(e)
            capture["done"] = True
        return out

    dct.collate = spy


# ---------------------------------------------------------------------------
# Ground-truth 2x2 grid (matches DreamTransform._prepare_video for non-droid):
#   top-left = top,  bottom-left = left_wrist,  top-right = right_wrist,
#   bottom-right = black.  Each view resized to (H, W) = (176, 320).
# ---------------------------------------------------------------------------

def build_gt_grid(dataset: AdamDataset, idx: int, H: int = 176, W: int = 320) -> np.ndarray:
    def view(server_key):
        f = dataset.get_frame(idx, server_key)  # (h, w, 3) RGB uint8
        return cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)

    top = view("video.top")
    left = view("video.left_wrist")
    right = view("video.right_wrist")

    grid = np.zeros((2 * H, 2 * W, 3), dtype=np.uint8)
    grid[:H, :W] = top      # top-left
    grid[H:, :W] = left     # bottom-left
    grid[:H, W:] = right    # top-right
    # bottom-right stays black
    return grid


def save_mp4(path: str, frames, fps: int = 5):
    """Write RGB uint8 frames to an MP4 via cv2.VideoWriter (no imageio dependency)."""
    frames = list(frames)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        # Fallback container/codec.
        alt = path.replace(".mp4", ".avi")
        writer = cv2.VideoWriter(alt, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
        path = alt
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29501")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)

    prompt_capture: dict = {}
    install_prompt_spy(prompt_capture)

    print(f"Loading model from {args.model_path} ...")
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.ADAM,
        model_path=args.model_path,
        device=args.device,
    )
    print("Model loaded.")

    # The action head's `dit_step_mask` (a step-skip cache schedule) is hardcoded to
    # 16 entries. If we raise num_inference_steps (WAM_NUM_INFERENCE_STEPS) above 16,
    # should_run_model() would IndexError. For a quality test we want every denoising
    # step to actually run the DiT, so size the mask to the step count (all-True).
    ah = policy.trained_model.action_head
    n_steps = int(getattr(ah, "num_inference_steps", len(ah.dit_step_mask)))
    if len(ah.dit_step_mask) != n_steps:
        ah.dit_step_mask = [True] * n_steps
        print(f"[cfg] resized dit_step_mask -> {n_steps} all-True (run every step)")
    print(f"[cfg] num_inference_steps={n_steps} cfg_scale={getattr(ah, 'cfg_scale', '?')} "
          f"seed={getattr(ah, 'seed', '?')}")

    dataset = AdamDataset(args.dataset_path)
    os.makedirs(args.output_dir, exist_ok=True)

    num = min(args.num_samples, len(dataset))
    video_across_time = []
    gt_grids = []
    used_prompt = None

    print(f"\nGenerating dynamics video over {num} samples "
          f"(start={args.start_idx}, stride={args.stride}) ...")
    print("-" * 60)

    for i in range(num):
        idx = args.start_idx + i * args.stride
        if idx >= len(dataset):
            print(f"Reached end of dataset at i={i}, idx={idx}")
            break

        prompt = args.prompt
        if args.use_dataset_prompt:
            task = dataset.get_task(idx)
            if task:
                prompt = task
        if used_prompt is None:
            used_prompt = prompt

        obs = build_obs(dataset, idx, prompt)
        gt_grids.append(build_gt_grid(dataset, idx))

        t0 = time.perf_counter()
        with torch.inference_mode():
            result, video_pred = policy.lazy_joint_forward_causal(Batch(obs=obs))
        elapsed = time.perf_counter() - t0

        # video_pred: (B, C=16, T, H_lat, W_lat) latent. Keep on GPU, detach.
        video_across_time.append(video_pred.detach())

        if i % args.log_every == 0:
            print(f"  [{i:>4d}/{num}] idx={idx} infer={elapsed:.3f}s "
                  f"video_pred={tuple(video_pred.shape)} prompt={prompt!r:.50}")

    if not video_across_time:
        print("No video predictions produced!")
        return

    # ---- Prompt verification ----
    print("\n" + "=" * 60)
    print("PROMPT CHECK")
    print(f"  raw task / --prompt : {used_prompt!r}")
    if prompt_capture.get("decoded"):
        print(f"  embodiment_id fed   : {prompt_capture.get('embodiment_id')} "
              f"(expect 32 = adam->YAM text branch)")
        print("  exact text to model (decoded from token ids):")
        for t in prompt_capture["decoded"]:
            print(f"    >> {t}")
    else:
        print(f"  (prompt spy did not capture: {prompt_capture.get('error')})")
    print("=" * 60)

    # ---- Decode latent video -> pixels ----
    ah = policy.trained_model.action_head
    latents = torch.cat(video_across_time, dim=2)
    print(f"\nDecoding latent video {tuple(latents.shape)} via VAE (tiled={ah.tiled}) ...")

    def decode(lat):
        with torch.inference_mode():
            return ah.vae.decode(
                lat,
                tiled=ah.tiled,
                tile_size=(ah.tile_size_height, ah.tile_size_width),
                tile_stride=(ah.tile_stride_height, ah.tile_stride_width),
            )

    try:
        frames = decode(latents)
    except RuntimeError as e:
        # Temporal chunked fallback if a single decode OOMs.
        print(f"  full decode failed ({e}); decoding in temporal chunks ...")
        torch.cuda.empty_cache()
        chunk = max(3, args.decode_chunk)
        parts = [decode(latents[:, :, s:s + chunk]) for s in range(0, latents.shape[2], chunk)]
        frames = torch.cat(parts, dim=2)

    frames = rearrange(frames, "B C T H W -> B T H W C")[0]
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    pred_path = save_mp4(os.path.join(args.output_dir, "pred_dynamic.mp4"), frames, fps=args.fps)
    print(f"  wrote {pred_path}  ({len(frames)} frames, {frames.shape[1]}x{frames.shape[2]})")

    # ---- GT grid reference video ----
    gt_path = save_mp4(os.path.join(args.output_dir, "gt_grid.mp4"), gt_grids, fps=args.fps)
    print(f"  wrote {gt_path}  ({len(gt_grids)} frames, {gt_grids[0].shape[0]}x{gt_grids[0].shape[1]})")

    # ---- Save a few still PNGs for quick visual inspection without a player ----
    n_still = min(6, len(frames))
    sel = np.linspace(0, len(frames) - 1, n_still).astype(int)
    for j, fi in enumerate(sel):
        cv2.imwrite(os.path.join(args.output_dir, f"pred_still_{j}_f{fi}.png"),
                    cv2.cvtColor(frames[fi], cv2.COLOR_RGB2BGR))

    # ---- Prompt + stats text ----
    with open(os.path.join(args.output_dir, "prompt.txt"), "w") as f:
        f.write(f"raw_task\t{used_prompt}\n")
        f.write(f"embodiment_id\t{prompt_capture.get('embodiment_id')}\n")
        for t in (prompt_capture.get("decoded") or []):
            f.write(f"decoded_prompt\t{t}\n")

    # ---- Cheap sanity stats on the predicted pixels ----
    fmean = float(frames.mean())
    fstd = float(frames.std())
    # temporal variation: mean abs diff between consecutive decoded frames
    tvar = float(np.mean(np.abs(np.diff(frames.astype(np.int16), axis=0)))) if len(frames) > 1 else 0.0
    print("\nPRED PIXEL STATS  (sanity):")
    print(f"  mean={fmean:.1f} std={fstd:.1f}  temporal_mean_abs_diff={tvar:.2f}")
    print("  (std≈0 or tvar≈0 => static/blank => conditioning bug; "
          "healthy std and nonzero tvar => real imagery)")
    print(f"\nResults in {os.path.abspath(args.output_dir)}/")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_path", default="./checkpoints/adam_stage_a_lora/checkpoint-10000")
    p.add_argument("--dataset_path", default="./data")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--prompt", default="Pick up the yellow straw and place it into the cup.")
    p.add_argument("--use_dataset_prompt", action="store_true")
    p.add_argument("--num_samples", type=int, default=48)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--output_dir", default="results_dynamic_10000")
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--decode_chunk", type=int, default=24,
                   help="temporal chunk size (latent frames) for VAE decode fallback")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
