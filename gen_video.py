#!/usr/bin/env python3
"""Diagnostic: generate WAM world-model video rollouts for MULTIPLE episodes
of the original (in-distribution) dataset, using the LoRA checkpoint.

Loads the model once, then for each episode feeds its first frame (+ state +
task), runs the causal joint forward, decodes video_pred through the VAE, saves
a per-episode mp4, and builds one combined labeled grid (rows = episodes).

Run:
  CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node=1 gen_video.py
Env:
  EPS="0,18,40,60,85,103"  MODEL_PATH=...  DATA=data
"""
import os, datetime
os.environ.setdefault("ENABLE_DIT_CACHE", "true")
os.environ.setdefault("ATTENTION_BACKEND", "TE")

import numpy as np, cv2, torch
import torch.distributed as dist
import pandas as pd
from einops import rearrange
import imageio
from tianshou.data import Batch
from torch.distributed.device_mesh import init_device_mesh
from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

MODEL_PATH = os.environ.get("MODEL_PATH", "./checkpoints/adam_stage_a_lora/checkpoint-5000")
DATA = os.environ.get("DATA", "data")
EPS = [int(x) for x in os.environ.get("EPS", "0,18,40,60,85,103").split(",")]
OUT = "video_pred_output"
CAMS = ["top", "left_wrist", "right_wrist"]
torch._dynamo.config.recompile_limit = 800
FONT = cv2.FONT_HERSHEY_SIMPLEX


def init_mesh():
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())
    return init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("ip",))


def load_first(ep):
    vids = {}
    for k in CAMS:
        c = cv2.VideoCapture(f"{DATA}/videos/chunk-000/observation.images.{k}/episode_{ep:06d}.mp4")
        ok, img = c.read(); c.release()
        vids[k] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    df = pd.read_parquet(f"{DATA}/data/chunk-000/episode_{ep:06d}.parquet")
    state = np.asarray(df["observation.state"].iloc[0], dtype=np.float64).reshape(-1)
    task = str(df["annotation.task"].iloc[0])
    return vids, state, task


def build_obs(vids, state, task):
    f = lambda x: x[np.newaxis].astype(np.uint8)
    return {
        "video.top": f(vids["top"]), "video.left_wrist": f(vids["left_wrist"]),
        "video.right_wrist": f(vids["right_wrist"]),
        "state.left_joint_pos": state[0:6].reshape(1, 6), "state.left_gripper_pos": state[6:7].reshape(1, 1),
        "state.right_joint_pos": state[7:13].reshape(1, 6), "state.right_gripper_pos": state[13:14].reshape(1, 1),
        "annotation.task": task,
    }


def decode(ah, video_pred):
    with torch.no_grad():
        frames = ah.vae.decode(video_pred, tiled=ah.tiled,
            tile_size=(ah.tile_size_height, ah.tile_size_width),
            tile_stride=(ah.tile_stride_height, ah.tile_stride_width))
    frames = rearrange(frames, "B C T H W -> B T H W C")[0]
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)  # RGB
    return frames


def row_image(frames_rgb, ep, task):
    n = len(frames_rgb)
    idxs = sorted(set(int(x) for x in np.linspace(0, n - 1, 5)))
    cells = []
    for k in idxs:
        c = cv2.cvtColor(cv2.resize(frames_rgb[k], (320, 176)), cv2.COLOR_RGB2BGR)
        cv2.rectangle(c, (0, 0), (70, 18), (0, 0, 0), -1)
        cv2.putText(c, f"t={k}", (4, 14), FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cells.append(c)
    row = np.concatenate(cells, axis=1)
    strip = np.zeros((24, row.shape[1], 3), np.uint8)
    cv2.putText(strip, f"ep{ep}: {task[:95]}", (4, 17), FONT, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
    return np.concatenate([strip, row], axis=0)


def main():
    mesh = init_mesh(); rank = dist.get_rank()
    print(f"[rank{rank}] loading {MODEL_PATH} ...", flush=True)
    policy = GrootSimPolicy(embodiment_tag=EmbodimentTag("adam"), model_path=MODEL_PATH,
                            device="cuda", device_mesh=mesh)
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for ep in EPS:
        try:
            vids, state, task = load_first(ep)
            prompt = task
            if os.environ.get("DESCRIBE", "0") == "1":
                t = task.lower()
                prompt = ("A multi-view video shows that a robot " + t +
                          " The video is split into four views: The top-left view shows the top camera,"
                          " the top-right view shows the right camera, the bottom-left view shows the left camera,"
                          " and the bottom-right view is a black screen. The robot " + t)
            obs = build_obs(vids, state, prompt)
            with torch.no_grad():
                _, video_pred = policy.lazy_joint_forward_causal(Batch(obs=obs))
            if video_pred is None:
                print(f"[rank{rank}] ep{ep}: video_pred is None, skipping", flush=True); continue
            frames = decode(policy.trained_model.action_head, video_pred)
            if rank == 0:
                imageio.mimsave(f"{OUT}/pred_ep{ep}.mp4", list(frames), fps=5, codec="libx264")
                rows.append(row_image(frames, ep, task))
                print(f"[rank0] ep{ep}: {len(frames)} frames  task={task!r}", flush=True)
        except Exception as e:
            import traceback as tb
            print(f"[rank{rank}] ep{ep} FAILED: {e}", flush=True); tb.print_exc()
        torch.cuda.empty_cache()
    if rank == 0 and rows:
        grid = np.concatenate(rows, axis=0)
        ts = datetime.datetime.now().strftime("%H%M%S")
        out = f"{OUT}/multi_pred_grid_{ts}.png"
        cv2.imwrite(out, grid); print(f"[rank0] SAVED grid {out} {grid.shape}", flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
