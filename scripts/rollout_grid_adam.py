#!/usr/bin/env python3
"""Open-loop world-model rollouts for MULTIPLE tasks, sampled at several time stages.

Primes once on each task's first frame, then autoregressively generates 2-latent-frame
blocks feeding the model's own latents back (no per-call reset; needs the rollout_no_reset
guard in WANPolicyHead). Decodes each rollout continuously (no seams), saves a per-task mp4,
and builds one grid:  rows = tasks,  columns = time stages (t = 0,4,8,12,16 s).

Horizon ~21 latent frames (~80 frames ~= 16s @ 5fps); past ~9 latent frames the I2V
conditioning is exhausted, so expect drift (weak dynamics head).

  CUDA_VISIBLE_DEVICES=0 python scripts/rollout_grid_adam.py \
     --model_path ./checkpoints/adam_stage_a_merged_19000 --dataset_path ./data \
     --device cuda:0 --n_tasks 6 --blocks 9 --output_dir results_rollout_grid_19000
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

FONT = cv2.FONT_HERSHEY_SIMPLEX


def pick_distinct_task_starts(ds, n_tasks):
    """Return [(global_start_index, task_str)] for the first episode of each distinct task."""
    out, seen = [], set()
    for e in range(len(ds.episodes)):
        start = ds.cum_lengths[e]
        task = ds.get_task(start)
        if task and task not in seen:
            seen.add(task); out.append((start, task))
            if len(out) >= n_tasks:
                break
    return out


def save_mp4(path, frames_rgb, fps=5):
    frames = list(frames_rgb)
    if not frames:
        return
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def rollout_one(policy, ds, start_idx, prompt, blocks):
    ah = policy.trained_model.action_head
    def fresh():
        return Batch(obs=build_obs(ds, start_idx, prompt))
    ah.rollout_no_reset = False
    with torch.inference_mode():
        _, acc = policy.lazy_joint_forward_causal(fresh())          # (B,C,3,H,W)
    ah.rollout_no_reset = True
    for _ in range(blocks):
        with torch.inference_mode():
            _, blk = policy.lazy_joint_forward_causal(fresh(), latent_video=acc.detach())  # (B,C,2,H,W)
        acc = torch.cat([acc, blk], dim=2)
    ah.rollout_no_reset = False
    with torch.inference_mode():
        fr = ah.vae.decode(acc, tiled=ah.tiled,
                           tile_size=(ah.tile_size_height, ah.tile_size_width),
                           tile_stride=(ah.tile_stride_height, ah.tile_stride_width))
    fr = rearrange(fr, "B C T H W -> B T H W C")[0]
    return ((fr.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)  # (N,H,W,3) RGB


def run(args):
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost"); os.environ.setdefault("MASTER_PORT", "29521")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
    policy = GrootSimPolicy(embodiment_tag=EmbodimentTag.ADAM, model_path=args.model_path, device=args.device)
    ah = policy.trained_model.action_head
    if len(ah.dit_step_mask) != int(getattr(ah, "num_inference_steps", 16)):
        ah.dit_step_mask = [True] * int(ah.num_inference_steps)
    ds = AdamDataset(args.dataset_path); os.makedirs(args.output_dir, exist_ok=True)

    tasks = pick_distinct_task_starts(ds, args.n_tasks)
    print(f"[grid] {len(tasks)} distinct tasks; blocks={args.blocks}")
    stages_s = [0, 4, 8, 12, 16]
    grid_rows = []
    for ti, (start, task) in enumerate(tasks):
        print(f"[grid] task {ti+1}/{len(tasks)} (ep_start={start}): {task!r}")
        try:
            frames = rollout_one(policy, ds, start, task, args.blocks)
        except Exception as e:
            import traceback; print(f"  FAILED: {e}"); traceback.print_exc(); continue
        N = len(frames)
        save_mp4(os.path.join(args.output_dir, f"task{ti}_rollout.mp4"), frames, args.fps)
        # build a labeled row: one cell per time stage
        cells = []
        for s in stages_s:
            fi = min(N - 1, s * args.fps)
            c = cv2.cvtColor(cv2.resize(frames[fi], (240, 132)), cv2.COLOR_RGB2BGR)
            cv2.rectangle(c, (0, 0), (54, 16), (0, 0, 0), -1)
            cv2.putText(c, f"{s}s", (4, 13), FONT, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            cells.append(c)
        row = np.concatenate(cells, axis=1)
        strip = np.zeros((20, row.shape[1], 3), np.uint8)
        cv2.putText(strip, f"ep{start}: {task[:90]}", (4, 15), FONT, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        grid_rows.append(np.concatenate([strip, row], axis=0))
        # quick drift readout: bottom-right (should stay ~0) at each stage
        brs = []
        for s in stages_s:
            fi = min(N - 1, s * args.fps); g = cv2.cvtColor(frames[fi], cv2.COLOR_RGB2GRAY).astype(np.float32)
            brs.append(int(g[g.shape[0]//2:, g.shape[1]//2:].mean()))
        print(f"    decoded {N} frames; BR(should~0) by stage {stages_s} = {brs}")
    if grid_rows:
        grid = np.concatenate(grid_rows, axis=0)
        out = os.path.join(args.output_dir, "tasks_x_stages_grid.png")
        cv2.imwrite(out, grid)
        print(f"[grid] SAVED {out}  {grid.shape}  (rows=tasks, cols=0/4/8/12/16s)")
    print(f"[grid] per-task mp4s + grid in {os.path.abspath(args.output_dir)}/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="./checkpoints/adam_stage_a_merged_19000")
    p.add_argument("--dataset_path", default="./data")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_tasks", type=int, default=6)
    p.add_argument("--blocks", type=int, default=9)
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--output_dir", default="results_rollout_grid_19000")
    run(p.parse_args())


if __name__ == "__main__":
    main()
