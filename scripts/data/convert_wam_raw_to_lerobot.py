"""
Convert RAW WAM teleop collection (ZED .svo2 + collect_adam.json) -> LeRobot v2.

Raw layout (one dir per session, named by timestamp):
  WAM/<YYYYMMDD_HHMMSS>/
    collect_adam.json     {"left":[frame,...], "right":[frame,...]}  proprioception
        frame = {timestamp, gripper:{pos,...}, arm:{joints[6],...}, ...}
    head_camera.svo2  left_camera.svo2  right_camera.svo2           ZED recordings
    task_info.json        {"task_description": "...", ...}

Target (LeRobot v2, matches data_merged so convert_lerobot_to_gear.py can finish it):
  observation.state / action : float32 [14] = [L_joints(6), L_gripper.pos, R_joints(6), R_gripper.pos]
  action[t] = state[t+1]      (next-state; verified exact against data_merged)
  videos                      : top<-head, left_wrist<-left, right_wrist<-right  (640x360)
  annotation.task             : task_info.task_description
  fps                         : 30, timestamp relative to episode start

Camera frames define the canonical timeline; proprioception is sampled (nearest) at each
camera-frame timestamp. With --no-video, a synthetic 30fps timeline over the proprio
overlap is used instead (lets the state/action half be built/tested without the ZED SDK).

ZED SDK note: decoding .svo2 REQUIRES Stereolabs `pyzed` (CUDA). If it's missing, run with
--no-video to produce parquet+meta only, or install the SDK first.

Usage:
  python scripts/data/convert_wam_raw_to_lerobot.py \
      --raw-root  /home/thematrix/tony/dreamzero/wam_work/WAM \
      --output    /home/thematrix/tony/dreamzero/wam_lerobot \
      --fps 30
  # state/action only (no ZED SDK), e.g. for validation:
  python scripts/data/convert_wam_raw_to_lerobot.py --raw-root ... --output ... --no-video --max-sessions 1
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("wam2lerobot")

VIDEO_W, VIDEO_H = 640, 360
CAM_MAP = {"top": "head_camera.svo2", "left_wrist": "left_camera.svo2", "right_wrist": "right_camera.svo2"}


def arm7(frame: dict) -> list[float]:
    """[joint0..5, gripper.pos] for one arm at one timestamp."""
    return list(frame["arm"]["joints"][:6]) + [float(frame["gripper"]["pos"])]


def stream(side_frames: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps[N], proprio[N,7]) sorted by time."""
    ts = np.array([f["timestamp"] for f in side_frames], dtype=np.float64)
    vals = np.array([arm7(f) for f in side_frames], dtype=np.float64)
    order = np.argsort(ts)
    return ts[order], vals[order]


def nearest(ts_src: np.ndarray, vals_src: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """Nearest-neighbor sample of vals_src(ts_src) at t_query."""
    idx = np.searchsorted(ts_src, t_query)
    idx = np.clip(idx, 1, len(ts_src) - 1)
    left = ts_src[idx - 1]
    right = ts_src[idx]
    pick_left = (t_query - left) <= (right - t_query)
    chosen = np.where(pick_left, idx - 1, idx)
    return vals_src[chosen]


def decode_svo(svo_path: Path):
    """Decode a ZED .svo2 -> (frames[list HxWx3 BGR uint8], timestamps_ns[np.int64]). Needs pyzed."""
    import pyzed.sl as sl  # noqa: import-outside-toplevel (optional heavy dep)
    cam = sl.Camera()
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED failed to open {svo_path}")
    rt = sl.RuntimeParameters()
    mat = sl.Mat()
    frames, ts = [], []
    while cam.grab(rt) == sl.ERROR_CODE.SUCCESS:
        cam.retrieve_image(mat, sl.VIEW.LEFT)
        bgr = mat.get_data()[:, :, :3].copy()  # BGRA -> BGR
        frames.append(bgr)
        ts.append(cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
    cam.close()
    return frames, np.array(ts, dtype=np.int64)


def write_mp4(frames, path: Path, fps: int):
    import imageio.v2 as imageio  # ffmpeg backend
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(str(path), fps=fps, codec="mpeg4", quality=8, macro_block_size=None)
    for f in frames:
        rgb = cv2.cvtColor(cv2.resize(f, (VIDEO_W, VIDEO_H)), cv2.COLOR_BGR2RGB)
        w.append_data(rgb)
    w.close()


def convert_session(sess: Path, ep_idx: int, out: Path, fps: int, no_video: bool, index_offset: int) -> dict:
    collect = json.load(open(sess / "collect_adam.json"))
    task = json.load(open(sess / "task_info.json")).get("task_description", "")
    lt, lv = stream(collect["left"])
    rt, rv = stream(collect["right"])

    if no_video:
        t0 = max(lt[0], rt[0]); t1 = min(lt[-1], rt[-1])
        timeline = np.arange(t0, t1, 1.0 / fps)
        cam_frames = None
    else:
        cam_frames, cam_ts = {}, {}
        for name, fn in CAM_MAP.items():
            fr, ts_ns = decode_svo(sess / fn)
            cam_frames[name] = fr
            cam_ts[name] = ts_ns.astype(np.float64) / 1e9  # ns -> s (epoch)
        ref = "top"
        timeline = cam_ts[ref]
        n = min(len(cam_frames[c]) for c in CAM_MAP)
        timeline = timeline[:n]
        for c in CAM_MAP:
            cam_frames[c] = cam_frames[c][:n]

    L = nearest(lt, lv, timeline)   # [N,7]
    R = nearest(rt, rv, timeline)   # [N,7]
    state = np.concatenate([L, R], axis=1).astype(np.float32)   # [N,14]
    action = np.empty_like(state)
    action[:-1] = state[1:]
    action[-1] = state[-1]
    N = len(state)

    df = pd.DataFrame({
        "frame_index": np.arange(N, dtype=np.int64),
        "episode_index": np.full(N, ep_idx, dtype=np.int64),
        "timestamp": (np.arange(N, dtype=np.float32) / fps),
        "observation.state": list(state),
        "action": list(action),
        "annotation.task": [task] * N,
        "index": np.arange(index_offset, index_offset + N, dtype=np.int64),
    })
    pq = out / f"data/chunk-000/episode_{ep_idx:06d}.parquet"
    pq.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(pq)

    if not no_video:
        for name in CAM_MAP:
            write_mp4(cam_frames[name], out / f"videos/chunk-000/observation.images.{name}/episode_{ep_idx:06d}.mp4", fps)

    log.info("  ep%04d %-16s N=%d task=%r", ep_idx, sess.name, N, task[:40])
    return {"episode_index": ep_idx, "tasks": [task], "length": N}


def write_info(out: Path, fps: int, total_frames: int, n_ep: int, with_video: bool):
    feats = {
        "observation.state": {"dtype": "float32", "shape": [14], "names": None},
        "action": {"dtype": "float32", "shape": [14], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "annotation.task": {"dtype": "string"},
    }
    if with_video:
        for name in CAM_MAP:
            feats[f"observation.images.{name}"] = {
                "dtype": "video", "shape": [VIDEO_H, VIDEO_W, 3],
                "names": ["height", "width", "channels"],
                "info": {"video.height": VIDEO_H, "video.width": VIDEO_W, "video.fps": fps,
                         "video.codec": "mpeg4", "video.pix_fmt": "yuv420p",
                         "video.is_depth_map": False, "video.channels": 3, "has_audio": False},
            }
    info = {
        "codebase_version": "v2.0", "robot_type": "adam",
        "total_episodes": n_ep, "total_frames": total_frames,
        "total_videos": n_ep * len(CAM_MAP) if with_video else 0,
        "total_chunks": 1, "total_tasks": None, "chunks_size": 1000, "fps": fps,
        "splits": {"train": f"0:{n_ep}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": feats,
    }
    (out / "meta").mkdir(parents=True, exist_ok=True)
    json.dump(info, open(out / "meta/info.json", "w"), indent=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True, help="dir containing the per-session timestamp folders")
    ap.add_argument("--output", required=True, help="output LeRobot v2 dataset dir")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-video", action="store_true", help="skip svo2 decode (no ZED SDK); parquet+meta only")
    ap.add_argument("--max-sessions", type=int, default=None, help="limit sessions (for testing)")
    args = ap.parse_args()

    raw = Path(args.raw_root); out = Path(args.output)
    sessions = sorted(p for p in raw.iterdir() if p.is_dir() and (p / "collect_adam.json").exists())
    if args.max_sessions:
        sessions = sessions[: args.max_sessions]
    log.info("found %d sessions under %s", len(sessions), raw)

    episodes, total = [], 0
    for i, sess in enumerate(sessions):
        ep = convert_session(sess, i, out, args.fps, args.no_video, index_offset=total)
        episodes.append(ep); total += ep["length"]

    # episodes.jsonl + tasks.jsonl
    (out / "meta").mkdir(parents=True, exist_ok=True)
    with open(out / "meta/episodes.jsonl", "w") as f:
        for e in episodes:
            f.write(json.dumps(e) + "\n")
    tasks = sorted({t for e in episodes for t in e["tasks"]})
    with open(out / "meta/tasks.jsonl", "w") as f:
        for idx, t in enumerate(tasks):
            f.write(json.dumps({"task_index": idx, "task": t}) + "\n")
    write_info(out, args.fps, total, len(episodes), with_video=not args.no_video)

    print("\n" + "=" * 60)
    print(f"LeRobot v2 written: {out}  ({len(episodes)} episodes, {total} frames)")
    print("Next: GEAR metadata ->")
    print(f"  python scripts/data/convert_lerobot_to_gear.py --dataset-path {out} \\")
    print("    --embodiment-tag adam \\")
    print("    --state-keys  '{\"left_joint_pos\":[0,6],\"left_gripper_pos\":[6,7],\"right_joint_pos\":[7,13],\"right_gripper_pos\":[13,14]}' \\")
    print("    --action-keys '{\"left_joint_pos\":[0,6],\"left_gripper_pos\":[6,7],\"right_joint_pos\":[7,13],\"right_gripper_pos\":[13,14]}' \\")
    print("    --relative-action-keys left_joint_pos left_gripper_pos right_joint_pos right_gripper_pos \\")
    print("    --task-key annotation.task")
    print("=" * 60)


if __name__ == "__main__":
    main()
