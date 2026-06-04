#!/usr/bin/env python3
"""Mimic real inference: re-anchor the world model on REAL frames sampled across the
WHOLE episode (closed-loop style), instead of open-loop rolling out from frame 0.

Deployment feeds a fresh real camera frame every control step; the model predicts a short
look-ahead from it, the robot moves, and the next real frame re-grounds it. That re-anchoring
is what keeps it on-task (approach -> grab -> place). Rolling out from only frame 0 drifts.

Here we sample N real frames spanning one full episode, run the normal (reset-every-call)
causal forward on each (exactly the inference path), accumulate video_pred, and decode. The
resulting video tracks the full task because each step is grounded in a real observation.

  CUDA_VISIBLE_DEVICES=0 python scripts/mimic_inference_adam.py \
     --model_path ./checkpoints/adam_stage_a_merged_19000 --dataset_path ./data \
     --device cuda:0 --episode 0 --steps 60 --output_dir results_mimic_infer_19000
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


def gt_grid(ds, idx, H=176, W=320):
    def v(k):
        return cv2.resize(ds.get_frame(idx, k), (W, H), interpolation=cv2.INTER_AREA)
    g = np.zeros((2 * H, 2 * W, 3), np.uint8)
    g[:H, :W] = v("video.top"); g[H:, :W] = v("video.left_wrist"); g[:H, W:] = v("video.right_wrist")
    return g  # RGB


def save_mp4(path, frames_rgb, fps=5):
    frames = list(frames_rgb)
    if not frames: return
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames: vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def run(args):
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost"); os.environ.setdefault("MASTER_PORT", "29522")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
    policy = GrootSimPolicy(embodiment_tag=EmbodimentTag.ADAM, model_path=args.model_path, device=args.device)
    ah = policy.trained_model.action_head
    if len(ah.dit_step_mask) != int(getattr(ah, "num_inference_steps", 16)):
        ah.dit_step_mask = [True] * int(ah.num_inference_steps)
    ds = AdamDataset(args.dataset_path); os.makedirs(args.output_dir, exist_ok=True)

    e = args.episode
    start = ds.cum_lengths[e]; length = ds.cum_lengths[e + 1] - start
    stride = max(1, length // args.steps)
    idxs = [start + i * stride for i in range(args.steps) if start + i * stride < start + length]
    task = ds.get_task(start)
    print(f"[mimic] episode {e}: len={length} frames (~{length/30:.1f}s), {len(idxs)} real-frame steps "
          f"(stride={stride}, spans full episode), task={task!r}")

    vids, gts = [], []
    for j, idx in enumerate(idxs):
        prompt = ds.get_task(idx) or task
        with torch.inference_mode():
            _, vp = policy.lazy_joint_forward_causal(Batch(obs=build_obs(ds, idx, prompt)))  # re-anchor each call
        vids.append(vp.detach()); gts.append(gt_grid(ds, idx))
        if j % 10 == 0: print(f"  step {j}/{len(idxs)} idx={idx}")

    acc = torch.cat(vids, dim=2)
    with torch.inference_mode():
        fr = ah.vae.decode(acc, tiled=ah.tiled, tile_size=(ah.tile_size_height, ah.tile_size_width),
                           tile_stride=(ah.tile_stride_height, ah.tile_stride_width))
    fr = rearrange(fr, "B C T H W -> B T H W C")[0]
    pred = ((fr.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    print(f"[mimic] decoded {len(pred)} pred frames")

    save_mp4(os.path.join(args.output_dir, "pred_mimic_infer.mp4"), pred, args.fps)
    save_mp4(os.path.join(args.output_dir, "gt_grid_fulltask.mp4"), gts, args.fps)

    # Comparison contact sheet: 5 task stages (0/25/50/75/100%), GT (top) vs PRED (bottom)
    pcts = [0.0, 0.25, 0.5, 0.75, 1.0]
    gt_cells, pr_cells = [], []
    for p in pcts:
        gi = min(len(gts) - 1, int(p * (len(gts) - 1)))
        pi = min(len(pred) - 1, int(p * (len(pred) - 1)))
        gc = cv2.cvtColor(cv2.resize(gts[gi], (240, 132)), cv2.COLOR_RGB2BGR)
        pc = cv2.cvtColor(cv2.resize(pred[pi], (240, 132)), cv2.COLOR_RGB2BGR)
        for img, lab in ((gc, f"GT {int(p*100)}%"), (pc, f"PRED {int(p*100)}%")):
            cv2.rectangle(img, (0, 0), (90, 16), (0, 0, 0), -1)
            cv2.putText(img, lab, (4, 13), FONT, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        gt_cells.append(gc); pr_cells.append(pc)
    sheet = np.vstack([np.hstack(gt_cells), np.hstack(pr_cells)])
    cv2.imwrite(os.path.join(args.output_dir, "gt_vs_pred_stages.png"), sheet)
    print(f"[mimic] saved pred_mimic_infer.mp4, gt_grid_fulltask.mp4, gt_vs_pred_stages.png in "
          f"{os.path.abspath(args.output_dir)}/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="./checkpoints/adam_stage_a_merged_19000")
    p.add_argument("--dataset_path", default="./data")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--steps", type=int, default=60, help="real-frame re-anchor steps spanning the episode")
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--output_dir", default="results_mimic_infer_19000")
    run(p.parse_args())


if __name__ == "__main__":
    main()
