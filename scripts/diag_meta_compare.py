#!/usr/bin/env python
"""Compare GEAR metadata between two datasets to explain a transform shape mismatch."""
import json, os, sys

WG = sys.argv[1] if len(sys.argv) > 1 else "/gpfs/scrubbed/rrrd/dreamzero/wam_geer_us/meta"
DM = sys.argv[2] if len(sys.argv) > 2 else "/gpfs/scrubbed/rrrd/dreamzero/data_merged/meta"

def load(p):
    return json.load(open(p)) if os.path.exists(p) else None

for tag, root in [("WAM_GEER_US (crashes)", WG), ("DATA_MERGED (works)", DM)]:
    print("=" * 25, tag, "=" * 25)
    info = load(os.path.join(root, "info.json"))
    feats = (info or {}).get("features", {})
    print("  fps:", (info or {}).get("fps"), "| episodes:", (info or {}).get("total_episodes"))
    for fk, fv in feats.items():
        if "image" in fk or "video" in fk:
            print("  IMG", fk, "shape=", fv.get("shape"), "names=", fv.get("names"))
        if fk in ("observation.state", "action"):
            print("  VEC", fk, "shape=", fv.get("shape"))
    mod = load(os.path.join(root, "modality.json"))
    print("  modality.json sections:", list(mod.keys()) if mod else None)
    if mod:
        for sect in mod:
            entry = mod[sect]
            # summarize delta-index lengths / keys per modality
            print("   [%s]" % sect, json.dumps(entry)[:500])
    print()
