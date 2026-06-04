#!/usr/bin/env python3
"""Does the model predict the GRIPPER CLOSE at the grasp moment, on TRAINING data?

The robot (and the open-loop video) approach the cube but never close the gripper. Since
actions+video come from the same model, that's a model behavior. This checks the decisive
question on in-distribution data: at a pre-grasp frame of a demo that DOES grasp, does the
predicted 24-step gripper chunk drop toward 'closed' like the ground-truth demo?

  -> predicted gripper closes  => model CAN grasp; robot failure is deployment/OOD.
  -> predicted gripper stays open => training gap (grasp behavior not learned).

  CUDA_VISIBLE_DEVICES=0 python scripts/diag_gripper_pred.py \
     --model_path ./checkpoints/adam_stage_a_merged_19000 --dataset_path ./data --episode_index 5
"""
import torch._dynamo
torch._dynamo.config.disable = True

import argparse, glob, os, sys
import numpy as np, pyarrow.parquet as pq, torch
import torch.distributed as dist
from tianshou.data import Batch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from open_loop_adam import AdamDataset, build_obs  # noqa: E402
from groot.vla.data.schema import EmbodimentTag  # noqa: E402
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy  # noqa: E402


def run(args):
    # GT gripper trajectory from the parquet for the chosen episode_index
    pf = glob.glob(f"{args.dataset_path}/data/**/episode_{args.episode_index:06d}.parquet", recursive=True)[0]
    t = pq.read_table(pf)
    gt_act = np.array(t.column("action").to_pylist())   # (T,14)
    task = str(t.column("annotation.task")[0].as_py())
    Lg = gt_act[:, 6]
    onset = next((i for i in range(len(Lg)) if Lg[i] < 70), int(np.argmin(Lg)))  # first 'closing' frame
    print(f"episode {args.episode_index}: {task!r}  rows={len(Lg)}  L-grip {Lg[0]:.0f}->min {Lg.min():.0f}; close onset ~frame {onset}")

    # map episode_index -> global start index in AdamDataset
    ds = AdamDataset(args.dataset_path)
    e = ds.episode_id.index(args.episode_index)
    start = ds.cum_lengths[e]

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost"); os.environ.setdefault("MASTER_PORT", "29540")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
    policy = GrootSimPolicy(embodiment_tag=EmbodimentTag.ADAM, model_path=args.model_path, device=args.device)

    print(f"\n{'frame':>6} {'GT_grip[0->minInChunk]':>24} {'PRED_grip[0->minInChunk]':>26}  verdict")
    for f in [max(0, onset - 20), max(0, onset - 10), onset, onset + 10]:
        if f + 24 > len(Lg):
            continue
        gt_chunk = Lg[f:f + 24]
        obs = build_obs(ds, start + f, task)
        with torch.inference_mode():
            result, _ = policy.lazy_joint_forward_causal(Batch(obs=obs))
        pv = result.act["action.left_gripper_pos"]
        pv = pv.cpu().numpy() if isinstance(pv, torch.Tensor) else np.asarray(pv)
        pred_chunk = pv.reshape(-1)[:24]
        gt_closes = gt_chunk.min() < 60
        pred_closes = pred_chunk.min() < 60
        verdict = ("GT closes & PRED closes" if (gt_closes and pred_closes)
                   else "GT closes, PRED STAYS OPEN  <-- training gap" if gt_closes
                   else "GT doesn't close here")
        print(f"{f:>6} {gt_chunk[0]:>10.0f}->{gt_chunk.min():>10.0f}    {pred_chunk[0]:>12.0f}->{pred_chunk.min():>10.0f}   {verdict}")
    print("\n(gripper units raw/10: ~84=open, ~28=closed. min<60 => the chunk commands a close.)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="./checkpoints/adam_stage_a_merged_19000")
    p.add_argument("--dataset_path", default="./data")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--episode_index", type=int, default=5)
    run(p.parse_args())


if __name__ == "__main__":
    main()
