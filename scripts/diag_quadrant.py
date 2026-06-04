#!/usr/bin/env python3
"""Diagnose the quadrant anomaly: top-left -> black, bottom-right (always-black input)
-> generates content, in the predicted frames.

For each of the first N dataset frames we run ONE causal forward call and decode that
call's video_pred IN ISOLATION (no cross-cliplet stitching / temporal-VAE bleed). We
then report per-decoded-frame quadrant brightness so we can see, within a single clean
prediction, whether the conditioning frame is correct and the PREDICTED frames flip the
layout. Stills are saved so we can tell rotation (upside-down scene in BR) from
hallucination.

Run:
  CUDA_VISIBLE_DEVICES=2 python scripts/diag_quadrant.py \
     --model_path ./checkpoints/adam_stage_a_lora/checkpoint-10000 \
     --dataset_path ./data --device cuda:0 --num_calls 4 --output_dir results_diag_quad
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


def quad(img):
    H, W = img.shape[:2]; h, w = H // 2, W // 2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return (g[:h, :w].mean(), g[:h, w:].mean(), g[h:, :w].mean(), g[h:, w:].mean())


def to_pixels(latent, ah):
    with torch.inference_mode():
        fr = ah.vae.decode(latent, tiled=ah.tiled,
                           tile_size=(ah.tile_size_height, ah.tile_size_width),
                           tile_stride=(ah.tile_stride_height, ah.tile_stride_width))
    fr = rearrange(fr, "B C T H W -> B T H W C")[0]
    fr = ((fr.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    return fr  # (T,H,W,3) RGB


def run(args):
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost"); os.environ.setdefault("MASTER_PORT", "29511")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
    policy = GrootSimPolicy(embodiment_tag=EmbodimentTag.ADAM, model_path=args.model_path, device=args.device)
    ah = policy.trained_model.action_head
    if len(ah.dit_step_mask) != int(getattr(ah, "num_inference_steps", 16)):
        ah.dit_step_mask = [True] * int(ah.num_inference_steps)
    ds = AdamDataset(args.dataset_path); os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'call':>4} {'pixframe':>9}  {'TL':>6} {'TR':>6} {'BL':>6} {'BR':>6}   role")
    for c in range(args.num_calls):
        obs = build_obs(ds, c, ds.get_task(c) or "do the task")
        with torch.inference_mode():
            _, vpred = policy.lazy_joint_forward_causal(Batch(obs=obs))  # (B,16,T,44,80), T=3 (cond + 2 pred)
        # decode this call's latent ALONE
        frames = to_pixels(vpred.detach(), ah)
        T = len(frames)
        for t in range(T):
            tl, tr, bl, br = quad(cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR))
            role = "conditioning" if t == 0 else "PREDICTED"
            print(f"{c:>4} {t:>9}  {tl:>6.0f} {tr:>6.0f} {bl:>6.0f} {br:>6.0f}   {role}")
        # save first (cond) and last (most-predicted) stills, plus a 180-rotated overlay of last
        cv2.imwrite(os.path.join(args.output_dir, f"call{c}_cond_t0.png"),
                    cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
        last = frames[-1]
        cv2.imwrite(os.path.join(args.output_dir, f"call{c}_pred_t{T-1}.png"),
                    cv2.cvtColor(last, cv2.COLOR_RGB2BGR))
        # rotate the predicted frame 180 deg: if it matches the conditioning layout,
        # the BR content is the top-scene rotated (=> geometric flip), not hallucination.
        rot = cv2.rotate(last, cv2.ROTATE_180)
        cv2.imwrite(os.path.join(args.output_dir, f"call{c}_pred_t{T-1}_rot180.png"),
                    cv2.cvtColor(rot, cv2.COLOR_RGB2BGR))
    print(f"\nStills in {os.path.abspath(args.output_dir)}/  "
          f"(compare call*_cond_t0 vs call*_pred_t* and *_rot180)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="./checkpoints/adam_stage_a_lora/checkpoint-10000")
    p.add_argument("--dataset_path", default="./data")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num_calls", type=int, default=4)
    p.add_argument("--output_dir", default="results_diag_quad")
    run(p.parse_args())


if __name__ == "__main__":
    main()
