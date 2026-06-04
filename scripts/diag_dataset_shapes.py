#!/usr/bin/env python
"""Find the sample/key whose transformed shape differs from the rest (the np.stack
collate crash at dreamzero_cotrain.py:163). CPU-only: builds the real train dataset
from a saved resolved conf.yaml and iterates __getitem__, recording per-key shapes.

Usage: python diag_dataset_shapes.py <conf.yaml> [N]
"""
import os, sys, collections

os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29577")

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from hydra.utils import instantiate

conf_path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 500

if not dist.is_initialized():
    dist.init_process_group(backend="gloo", rank=0, world_size=1)

torch.manual_seed(42)
np.random.seed(42)

cfg = OmegaConf.load(conf_path)
print("instantiating train_dataset ...", flush=True)
ds = instantiate(cfg.train_dataset)
print(f"iterating first {N} samples (iterable dataset)", flush=True)

shape_counts = collections.defaultdict(collections.Counter)   # key -> Counter(shape)
idx_by_shape = collections.defaultdict(lambda: collections.defaultdict(list))  # key -> shape -> [idx]
errs = []

it = iter(ds)
for i in range(N):
    try:
        item = next(it)
    except StopIteration:
        print(f"StopIteration at {i}", flush=True)
        break
    except Exception as e:
        errs.append((i, repr(e)[:200]))
        continue
    for k, v in item.items():
        sh = tuple(v.shape) if hasattr(v, "shape") else None
        if sh is not None:
            shape_counts[k][sh] += 1
            idx_by_shape[k][sh].append(i)
    if (i + 1) % 100 == 0:
        print(f"  ...{i+1}", flush=True)

print("\n=== per-key shape distribution ===", flush=True)
for k, c in shape_counts.items():
    print(f"  {k}: {dict(c)}")

print("\n=== KEYS WITH >1 SHAPE (the culprit) ===", flush=True)
found = False
for k, c in shape_counts.items():
    if len(c) > 1:
        found = True
        majority = c.most_common(1)[0][0]
        print(f"  [{k}] majority={majority}")
        for sh, cnt in c.items():
            if sh != majority:
                print(f"      minority {sh} x{cnt} at indices {idx_by_shape[k][sh][:20]}")
if not found:
    print("  (all keys uniform in first N — bad sample is at a higher index or random-window-dependent)")

if errs:
    print(f"\n=== {len(errs)} __getitem__ errors ===")
    for i, e in errs[:10]:
        print(f"  idx {i}: {e}")
