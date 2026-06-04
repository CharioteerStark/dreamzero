#!/usr/bin/env python3
"""Ablations for the predicted-video degeneracy (TL->black, BR->content).

Experiment A (seed sweep): is the degenerate pattern noise-locked or structural?
  Run the same single forward call with several noise seeds; if the TL-black/BR-content
  pattern persists across seeds, it's structural (not just the fixed seed).

Experiment B (LoRA disabled): what does the *inference base* DiT do on its own?
  NOTE: at inference the base DiT is loaded from Wan2.1-I2V-14B-480P (raw), NOT the
  DreamZero-AgiBot checkpoint the LoRA was trained on. Disabling the PEFT adapter shows
  the raw-base behavior on Adam frames. If raw-base is already degenerate, the rank-4
  LoRA can't rescue it and the real issue is the wrong base at inference.

One model load; per-call decode in isolation (no stitching). Saves stills + a table.

  CUDA_VISIBLE_DEVICES=2 python scripts/diag_ablate.py \
      --model_path ./checkpoints/adam_stage_a_lora/checkpoint-10000 \
      --dataset_path ./data --device cuda:0 --output_dir results_ablate
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


def quad(rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]; h, w = H // 2, W // 2
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return (g[:h, :w].mean(), g[:h, w:].mean(), g[h:, :w].mean(), g[h:, w:].mean())


def decode(latent, ah):
    with torch.inference_mode():
        fr = ah.vae.decode(latent, tiled=ah.tiled,
                           tile_size=(ah.tile_size_height, ah.tile_size_width),
                           tile_stride=(ah.tile_stride_height, ah.tile_stride_width))
    fr = rearrange(fr, "B C T H W -> B T H W C")[0]
    return ((fr.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)


def one_call(policy, ds, idx, ah, outdir, tag):
    obs = build_obs(ds, idx, ds.get_task(idx) or "do the task")
    with torch.inference_mode():
        _, vpred = policy.lazy_joint_forward_causal(Batch(obs=obs))
    frames = decode(vpred.detach(), ah)
    c = quad(frames[0]); p = quad(frames[-1])
    print(f"{tag:<22} cond[TL,TR,BL,BR]=({c[0]:.0f},{c[1]:.0f},{c[2]:.0f},{c[3]:.0f})  "
          f"pred[TL,TR,BL,BR]=({p[0]:.0f},{p[1]:.0f},{p[2]:.0f},{p[3]:.0f})")
    cv2.imwrite(os.path.join(outdir, f"{tag}_pred.png"), cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))
    return c, p


def run(args):
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost"); os.environ.setdefault("MASTER_PORT", "29512")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
    policy = GrootSimPolicy(embodiment_tag=EmbodimentTag.ADAM, model_path=args.model_path, device=args.device)
    ah = policy.trained_model.action_head
    if len(ah.dit_step_mask) != int(getattr(ah, "num_inference_steps", 16)):
        ah.dit_step_mask = [True] * int(ah.num_inference_steps)
    ds = AdamDataset(args.dataset_path); os.makedirs(args.output_dir, exist_ok=True)
    idx = args.idx

    print("\n========== EXPERIMENT A: seed sweep (LoRA ENABLED) ==========")
    for s in [1140, 42, 777, 2024]:
        ah.seed = s
        one_call(policy, ds, idx, ah, args.output_dir, f"A_seed{s}")

    print("\n========== EXPERIMENT B: LoRA DISABLED (raw inference base = Wan2.1-I2V) ==========")
    ah.seed = 1140
    peft_model = ah.model
    if hasattr(peft_model, "disable_adapter"):
        with peft_model.disable_adapter():
            one_call(policy, ds, idx, ah, args.output_dir, "B_noLoRA_rawWan")
    else:
        print("  (could not find disable_adapter on ah.model; type=%s)" % type(peft_model))

    print("\n========== reference: LoRA ENABLED, seed 1140 ==========")
    ah.seed = 1140
    one_call(policy, ds, idx, ah, args.output_dir, "ref_LoRA_seed1140")
    print(f"\nStills in {os.path.abspath(args.output_dir)}/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="./checkpoints/adam_stage_a_lora/checkpoint-10000")
    p.add_argument("--dataset_path", default="./data")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--idx", type=int, default=0)
    p.add_argument("--output_dir", default="results_ablate")
    run(p.parse_args())


if __name__ == "__main__":
    main()
