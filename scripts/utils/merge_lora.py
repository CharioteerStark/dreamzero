"""Merge a DreamZero LoRA checkpoint into its dense base.

Produces a dense safetensors checkpoint suitable as a `pretrained_model_path`
(e.g. to fold an embodiment-adaptation LoRA back into a dense base before further
training or deployment).

This is a pure state-dict merge — no model instantiation, no GPU required.

Save format assumption (matches `save_lora_only=true` in this codebase):
  Stage A LoRA save contains:
    - <prefix>.lora_A.<adapter>.weight       (rank x in_features, fp32 or bf16)
    - <prefix>.lora_B.<adapter>.weight       (out_features x rank)
    - other non-LoRA trainable weights        (e.g. state_encoder / action_encoder /
                                               action_decoder under action_head.model.*)
  Base checkpoint contains:
    - <prefix>.weight                          (out_features x in_features)
  Merged output (this script):
    - <prefix>.weight  =  <prefix>.weight  +  (alpha / rank) * (lora_B @ lora_A)
    - non-LoRA trainable weights are overwritten from the LoRA save (last-writer-wins)
    - lora_A / lora_B keys are dropped

Usage:
  python scripts/utils/merge_lora.py \
      --base ./checkpoints/DreamZero-AgiBot \
      --lora ./checkpoints/adam_stage_a_lora/checkpoint-5000 \
      --output ./checkpoints/adam_stage_a_merged
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


LORA_A_RE = re.compile(r"^(?P<prefix>.+)\.lora_A\.(?P<adapter>[^.]+)\.weight$")


def load_shards(path: Path) -> dict[str, torch.Tensor]:
    """Load a single safetensors file or a sharded safetensors directory."""
    idx = path / "model.safetensors.index.json"
    one = path / "model.safetensors"
    state: dict[str, torch.Tensor] = {}
    if idx.exists():
        index = json.loads(idx.read_text())
        for shard in sorted(set(index["weight_map"].values())):
            state.update(load_file(str(path / shard)))
    elif one.exists():
        state.update(load_file(str(one)))
    else:
        raise FileNotFoundError(f"No model.safetensors[.index.json] under {path}")
    return state


def main():
    p = argparse.ArgumentParser(
        description="Merge a Stage A DreamZero LoRA into its dense base (CPU-only, state-dict arithmetic).",
    )
    p.add_argument("--base", required=True, help="Dense base checkpoint dir (e.g. ./checkpoints/DreamZero-AgiBot)")
    p.add_argument("--lora", required=True, help="LoRA checkpoint dir from Stage A training (save_lora_only=true)")
    p.add_argument("--output", required=True, help="Output dir for merged dense checkpoint")
    p.add_argument("--alpha", type=int, default=4,
                   help="LoRA alpha used during training (default 4; see wan_flow_matching_action_tf.yaml)")
    p.add_argument("--rank", type=int, default=4,
                   help="LoRA rank used during training (default 4)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite --output if it already exists and is non-empty")
    args = p.parse_args()

    base_dir = Path(args.base).resolve()
    lora_dir = Path(args.lora).resolve()
    out_dir = Path(args.output).resolve()

    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"--output {out_dir} exists and is non-empty. Pass --force to overwrite."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    scaling = args.alpha / args.rank
    print(f"alpha/rank = {args.alpha}/{args.rank} = scaling={scaling}")

    print(f"Loading base weights from {base_dir} ...")
    base_state = load_shards(base_dir)
    print(f"  base: {len(base_state)} tensors")

    print(f"Loading LoRA weights from {lora_dir} ...")
    lora_state = load_shards(lora_dir)
    print(f"  lora: {len(lora_state)} tensors")

    merged: dict[str, torch.Tensor] = {k: v for k, v in base_state.items()}
    pairs_merged = 0
    non_lora_overrides = 0
    missing_base = []
    orphan_lora_b = []

    used_keys: set[str] = set()

    # Pass 1: merge LoRA pairs.
    for k, v in lora_state.items():
        m = LORA_A_RE.match(k)
        if m is None:
            continue
        prefix = m.group("prefix")
        adapter = m.group("adapter")
        b_key = f"{prefix}.lora_B.{adapter}.weight"
        base_key = f"{prefix}.weight"
        if b_key not in lora_state:
            print(f"  WARN: lora_A without matching lora_B: {k}")
            used_keys.add(k)
            continue
        if base_key not in merged:
            missing_base.append(base_key)
            used_keys.add(k)
            used_keys.add(b_key)
            continue
        a = lora_state[k].to(torch.float32)
        b = lora_state[b_key].to(torch.float32)
        delta = (b @ a) * scaling
        dst_dtype = merged[base_key].dtype
        merged[base_key] = (merged[base_key].to(torch.float32) + delta).to(dst_dtype)
        used_keys.add(k)
        used_keys.add(b_key)
        pairs_merged += 1

    # Pass 2: overwrite non-LoRA trainable weights (e.g. action_head.model.state_encoder.*,
    # action_encoder.*, action_decoder.* — all set to requires_grad=True in
    # wan_flow_matching_action_tf.py and therefore present in the lora-only save).
    for k, v in lora_state.items():
        if k in used_keys:
            continue
        if ".lora_A." in k or ".lora_B." in k:
            orphan_lora_b.append(k)
            continue
        if k in merged:
            merged[k] = v
            non_lora_overrides += 1
        else:
            # Unknown extra key — keep it; downstream load_state_dict(strict=False)
            # will silently drop it if not consumed.
            merged[k] = v
            non_lora_overrides += 1

    # Drop any remaining lora_A/lora_B entries from the merged dict.
    merged = {k: v for k, v in merged.items() if ".lora_A." not in k and ".lora_B." not in k}

    print(f"  merged {pairs_merged} LoRA pairs into base weights")
    print(f"  overrode {non_lora_overrides} non-LoRA weights from LoRA save (encoders/decoders)")
    if missing_base:
        print(f"  WARN: {len(missing_base)} LoRA pairs had no matching base weight. "
              f"First few: {missing_base[:5]}")
    if orphan_lora_b:
        print(f"  WARN: {len(orphan_lora_b)} orphan lora_B keys (no matching lora_A): "
              f"{orphan_lora_b[:3]}...")

    out_path = out_dir / "model.safetensors"
    print(f"Saving merged checkpoint to {out_path} ...")
    save_file(merged, str(out_path))

    # Copy config.json from the base — Stage B uses skip_component_loading=true and
    # defer_lora_injection=true, so it will inject a fresh LoRA on top of these dense weights.
    base_config = base_dir / "config.json"
    if base_config.exists():
        (out_dir / "config.json").write_text(base_config.read_text())
        print("  copied config.json from base")
    else:
        print("  NOTE: no config.json found in base dir; Stage B expects one at the pretrained path")

    print("Done.")


if __name__ == "__main__":
    main()
