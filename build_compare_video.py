#!/usr/bin/env python3
"""Build a side-by-side [ACTUAL observed | PREDICTED world-model] video from a serve_wam
--save-video-dir run. Shows where the action model (reality, left) diverges from the video
model (prediction, right): compare PRED at replan i to ACTUAL at replan i+1.

Pairs actual_NNNN_*.png with replan_NNNN_*.mp4 by index.

  python build_compare_video.py --dir ./world_model_videos
"""
import argparse, glob, os, re
import cv2, numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img, text, color):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (170, 22), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 16), FONT, 0.5, color, 1, cv2.LINE_AA)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="./world_model_videos")
    p.add_argument("--out", default=None)
    p.add_argument("--fps", type=int, default=5)
    args = p.parse_args()
    out = args.out or os.path.join(args.dir, "actual_vs_predicted.mp4")

    def index_map(pattern, grp):
        d = {}
        for f in glob.glob(os.path.join(args.dir, pattern)):
            m = re.search(grp, os.path.basename(f))
            if m:
                d[int(m.group(1))] = f
        return d

    actuals = index_map("actual_*.png", r"actual_(\d+)_")
    preds = index_map("replan_*.mp4", r"replan_(\d+)_")
    idxs = sorted(set(actuals) & set(preds))
    if not idxs:
        print(f"No paired actual_*.png / replan_*.mp4 in {args.dir}"); return
    print(f"{len(idxs)} paired replans -> {out}")

    writer = None
    for i in idxs:
        actual = cv2.imread(actuals[i])
        cap = cv2.VideoCapture(preds[i]); pframes = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            pframes.append(fr)
        cap.release()
        if not pframes:
            continue
        H, W = pframes[0].shape[:2]
        al = _label(cv2.resize(actual, (W, H)), f"ACTUAL r{i}", (0, 255, 255))
        for pf in pframes:
            sbs = np.hstack([al, _label(pf, f"PRED r{i}", (0, 255, 0))])
            if writer is None:
                writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.fps, (sbs.shape[1], sbs.shape[0]))
            writer.write(sbs)
    if writer:
        writer.release()
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
