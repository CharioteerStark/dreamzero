#!/usr/bin/env python
"""Reproduce the collate np.stack crash on CPU (no model/GPU/wandb): build the real
train dataset + data_collator from a saved conf.yaml and iterate batches at the given
batch size until the instrumented collate (dreamzero_cotrain.py) prints COLLATE-DIAG.
Usage: python diag_collate_repro.py <conf.yaml> [batch_size=4] [max_batches=1000]
"""
import os, sys

os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29581")

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader

torch.manual_seed(42)
np.random.seed(42)

conf_path = sys.argv[1]
BS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
MAXB = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

if not dist.is_initialized():
    dist.init_process_group(backend="gloo", rank=0, world_size=1)

cfg = OmegaConf.load(conf_path)
print("instantiating dataset (shard caching takes a few minutes) ...", flush=True)
ds = instantiate(cfg.train_dataset)
collator = instantiate(cfg.data_collator)
dl = DataLoader(ds, batch_size=BS, collate_fn=collator, num_workers=0)
print(f"iterating batches bs={BS}, up to {MAXB} ...", flush=True)

i = -1
try:
    for i, batch in enumerate(dl):
        if i == 0:
            print("batch0 keys + shapes:", flush=True)
            for k, v in batch.items():
                print(f"   {k}: {tuple(v.shape) if hasattr(v,'shape') else type(v).__name__}", flush=True)
        if i % 25 == 0:
            print(f"  batch {i} ok", flush=True)
        if i >= MAXB:
            print(f"reached {MAXB} batches with NO crash at bs={BS}", flush=True)
            break
except Exception as e:
    print(f"\n!!! CRASH at batch {i} (bs={BS}): {type(e).__name__}: {e}", flush=True)
    raise
