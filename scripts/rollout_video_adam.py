#!/usr/bin/env python3
"""Open-loop autoregressive world-model rollout for Adam (one long video per start frame).

Unlike dump_dynamic_video_adam.py (which re-anchors on a fresh real frame every call ->
short cliplets with periodic seams), this primes ONCE on a start frame, then repeatedly
generates 2-latent-frame blocks feeding the model's OWN previous latents back (no reset),
so the video is one continuous rollout. Decodes once (no seams) and saves 4/8/16 s clips.

Requires the `rollout_no_reset` guard in WANPolicyHead (action head). The horizon is bounded
by the causal attention window (~21 latent frames ~= 80 decoded frames ~= 16 s @ 5 fps); past
~9 latent frames the I2V conditioning is exhausted so expect drift (weak dynamics head).

  CUDA_VISIBLE_DEVICES=0 python scripts/rollout_video_adam.py \
     --model_path ./checkpoints/adam_stage_a_merged_19000 --dataset_path ./data \
     --device cuda:0 --start_idx 0 --blocks 9 --output_dir results_rollout_19000
"""
import torch._dynamo
torch._dynamo.config.disable = True

import argparse, os, sys
import cv2, numpy as np, torch
import torch.distributed as dist
from einops import rearrange
from tianshou.data import Batch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from open_loop_adam import AdamDataset, build_obs  # noqa: E402
from groot.vla.data.schema import EmbodimentTag  # noqa: E402
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy  # noqa: E402


def save_mp4(path, frames_rgb, fps=5):
    frames = list(frames_rgb)
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    return path


def run(args):
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost"); os.environ.setdefault("MASTER_PORT", "29520")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
    policy = GrootSimPolicy(embodiment_tag=EmbodimentTag.ADAM, model_path=args.model_path, device=args.device)
    ah = policy.trained_model.action_head
    if len(ah.dit_step_mask) != int(getattr(ah, "num_inference_steps", 16)):
        ah.dit_step_mask = [True] * int(ah.num_inference_steps)
    ds = AdamDataset(args.dataset_path); os.makedirs(args.output_dir, exist_ok=True)

    prompt = args.prompt or (ds.get_task(args.start_idx) or "do the task")
    print(f"[rollout] start_idx={args.start_idx} blocks={args.blocks} prompt={prompt!r}")

    def fresh_batch():
        return Batch(obs=build_obs(ds, args.start_idx, prompt))

    # ---- prime: 1 conditioning + first 2-frame block (current_start_frame: 0 -> 3) ----
    ah.rollout_no_reset = False
    with torch.inference_mode():
        _, acc = policy.lazy_joint_forward_causal(fresh_batch())   # (B, C, 3, H, W)
    print(f"[rollout] primed: acc latent frames = {acc.shape[2]} (start_frame={ah.current_start_frame})")

    # ---- continuations: feed own latents back, no reset ----
    ah.rollout_no_reset = True
    for k in range(args.blocks):
        with torch.inference_mode():
            _, blk = policy.lazy_joint_forward_causal(fresh_batch(), latent_video=acc.detach(), video_only=True)
        acc = torch.cat([acc, blk], dim=2)
        print(f"[rollout] block {k+1}/{args.blocks}: +{blk.shape[2]} -> {acc.shape[2]} latent frames "
              f"(start_frame={ah.current_start_frame})")
    ah.rollout_no_reset = False

    # ---- decode the whole rollout ONCE (continuous, no seams) ----
    print(f"[rollout] decoding {acc.shape[2]} latent frames ...")
    with torch.inference_mode():
        frames = ah.vae.decode(acc, tiled=ah.tiled,
                               tile_size=(ah.tile_size_height, ah.tile_size_width),
                               tile_stride=(ah.tile_stride_height, ah.tile_stride_width))
    frames = rearrange(frames, "B C T H W -> B T H W C")[0]
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    N = len(frames)
    print(f"[rollout] decoded {N} pixel frames ({N/args.fps:.1f}s @ {args.fps}fps)")

    save_mp4(os.path.join(args.output_dir, "rollout_full.mp4"), frames, args.fps)
    for secs in (4, 8, 16):
        n = min(N, secs * args.fps)
        save_mp4(os.path.join(args.output_dir, f"rollout_{secs}s.mp4"), frames[:n], args.fps)
        # per-quadrant BR (should stay ~0) at the END of each clip = how far it held up
        end = frames[n - 1]; H, W = end.shape[:2]; g = cv2.cvtColor(end, cv2.COLOR_RGB2GRAY).astype(np.float32)
        br = g[H // 2:, W // 2:].mean()
        print(f"  {secs}s ({n} frames): last-frame BR(should~0)={br:.0f}  contrast={g.std():.0f}")

    # contact sheet across the full rollout
    sel = np.linspace(0, N - 1, 6).astype(int)
    cells = []
    for fi in sel:
        im = cv2.cvtColor(frames[fi], cv2.COLOR_RGB2BGR)
        cv2.rectangle(im, (0, 0), (im.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(im, f"t={fi/args.fps:.1f}s (f{fi})", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cells.append(im)
    rows = [np.hstack(cells[i:i + 2]) for i in range(0, len(cells), 2)]
    cv2.imwrite(os.path.join(args.output_dir, "contact_sheet.png"), np.vstack(rows))
    print(f"[rollout] saved clips + contact_sheet in {os.path.abspath(args.output_dir)}/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="./checkpoints/adam_stage_a_merged_19000")
    p.add_argument("--dataset_path", default="./data")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--blocks", type=int, default=9, help="continuation blocks (2 latent frames each); 9 -> ~21 latent ~= 16s")
    p.add_argument("--prompt", default=None, help="defaults to the start episode's dataset task")
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--output_dir", default="results_rollout_19000")
    run(p.parse_args())


if __name__ == "__main__":
    main()
