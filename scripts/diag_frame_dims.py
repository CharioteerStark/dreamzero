#!/usr/bin/env python
"""Exhaustively decode EVERY frame of EVERY video and report actual per-frame
dimensions. Metadata (cv2 CAP_PROP) only reports the declared size; this catches
videos that declare 640x360 but decode a frame at a different size (the np.stack
collate crash). Parallel across cores. Usage: python diag_frame_dims.py <root> [nproc]
"""
import glob, os, sys, collections
from multiprocessing import Pool
import cv2

def check(f):
    cap = cv2.VideoCapture(f)
    shapes = collections.Counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        shapes[(w, h)] += 1
    cap.release()
    return (f, dict(shapes))

if __name__ == "__main__":
    root = sys.argv[1]
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    files = sorted(glob.glob(root + "/videos/**/*.mp4", recursive=True))
    print(f"decoding {len(files)} videos with {nproc} procs ...", flush=True)
    with Pool(nproc) as p:
        results = p.map(check, files)

    overall = collections.Counter()
    nonuniform = []          # videos whose own frames have >1 size
    for f, shapes in results:
        for wh, c in shapes.items():
            overall[wh] += c
        if len(shapes) > 1:
            nonuniform.append((f, shapes))

    print("\n=== overall decoded-frame dim distribution (WxH: #frames) ===")
    for wh, c in overall.most_common():
        print(f"  {wh[0]}x{wh[1]}: {c}")

    mode = overall.most_common(1)[0][0]
    print(f"\nglobal mode = {mode[0]}x{mode[1]}")

    print(f"\n=== videos with INTERNALLY non-uniform frame sizes: {len(nonuniform)} ===")
    for f, shapes in nonuniform:
        v = os.path.basename(os.path.dirname(f)); ep = os.path.basename(f)
        print(f"  {v}/{ep}: {shapes}")

    print("\n=== videos that decode any frame != global mode ===")
    any_outlier = False
    for f, shapes in results:
        odd = {wh: c for wh, c in shapes.items() if wh != mode}
        if odd:
            any_outlier = True
            v = os.path.basename(os.path.dirname(f)); ep = os.path.basename(f)
            print(f"  OUTLIER {v}/{ep}: {odd}  (mode {mode[0]}x{mode[1]})")
    if not any_outlier:
        print("  none — every frame of every video decodes to the mode size")
