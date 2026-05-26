#!/usr/bin/env python3
"""Adam bi-manual real-robot deployment client.

Subscribes to three ZED camera ZMQ feeds (published by pub_zed.py on the
Jetson), reads joint+gripper state from both xArm 6 arms, sends observations
to serve_wam.py, and executes the returned 14-D actions on the arms.

State / action layout (matches training):
    [left_joint(6 rad), left_gripper(raw/10), right_joint(6 rad), right_gripper(raw/10)]

Defaults to --dry-run (logs targets, no motion). Pass --no-dry-run to move.

Usage:
    python deploy_adam.py --zmq-host <jetson_ip> --prompt "pick up the cube"
    python deploy_adam.py --zmq-host <jetson_ip> --no-dry-run --prompt "pick up the cube"
"""

import argparse
import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np
import zmq
from xarm.wrapper import XArmAPI

from eval_utils.action_chunk_broker import AsyncActionChunkBroker
from eval_utils.wam_client import WamClientPolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ZMQ camera subscriber
# ---------------------------------------------------------------------------

class _ZmqCamSubscriber:
    """Background thread that keeps the latest JPEG-decoded RGB frame in memory."""

    def __init__(self, name: str, host: str, port: int):
        self.name = name
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._stamp = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            args=(host, port),
            name=f"cam_{name}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self, host: str, port: int) -> None:
        ctx = zmq.Context.instance()
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        sub.setsockopt(zmq.CONFLATE, 1)   # always keep latest frame only
        sub.setsockopt(zmq.RCVTIMEO, 500)
        sub.setsockopt(zmq.LINGER, 0)
        sub.connect(f"tcp://{host}:{port}")
        while not self._stop.is_set():
            try:
                data = sub.recv()
            except zmq.error.Again:
                continue
            frame_bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._frame = frame_rgb
                self._stamp = time.time()
        sub.close()

    def get_latest(self, max_age_s: float) -> Optional[np.ndarray]:
        with self._lock:
            if self._frame is None or time.time() - self._stamp > max_age_s:
                return None
            return self._frame


# ---------------------------------------------------------------------------
# xArm helpers
# ---------------------------------------------------------------------------

def _prepare_arm(arm: XArmAPI, label: str) -> None:
    arm.motion_enable(enable=True)
    arm.clean_warn()
    arm.clean_error()
    time.sleep(0.2)
    # Mode 6: joint online planning — accepts sparse targets at ~10 Hz and
    # blends them with online trajectory generation.
    arm.set_mode(6)
    time.sleep(0.1)
    arm.set_state(state=0)
    time.sleep(0.1)
    code = arm.set_gripper_enable(True)
    if code != 0:
        logger.warning("%s set_gripper_enable returned %d (continuing)", label, code)
    time.sleep(0.1)
    logger.info("%s arm ready (mode=6 joint online planning)", label)


def _read_arm_state(arm: XArmAPI, label: str) -> tuple[list[float], float]:
    """Return (6 joint angles in radians, gripper in dataset units = raw/10)."""
    code, angles = arm.get_servo_angle(is_radian=True)
    if code != 0:
        raise RuntimeError(f"{label} get_servo_angle failed: code={code}")
    code, gripper_raw = arm.get_gripper_position()
    if code != 0:
        raise RuntimeError(f"{label} get_gripper_position failed: code={code}")
    return list(angles[:6]), float(gripper_raw) / 10.0


def _apply_arm_action(
    left_arm: XArmAPI,
    right_arm: XArmAPI,
    action: np.ndarray,
    speed_rad_s: float,
    acc_rad_s2: float,
) -> None:
    """Send 14-D action to both arms (non-blocking)."""
    left_joints  = action[0:6].tolist()
    left_grip    = float(action[6])
    right_joints = action[7:13].tolist()
    right_grip   = float(action[13])

    left_arm.set_servo_angle(
        angle=left_joints, is_radian=True, wait=False,
        speed=speed_rad_s, mvacc=acc_rad_s2,
    )
    right_arm.set_servo_angle(
        angle=right_joints, is_radian=True, wait=False,
        speed=speed_rad_s, mvacc=acc_rad_s2,
    )
    # Gripper: dataset stores raw/10; xArm expects raw int in [0, 840].
    # speed=1000 r/min is the default max; use a gentler value to avoid sudden jumps.
    left_arm.set_gripper_position(int(np.clip(left_grip * 10.0, 0, 840)), wait=False, speed=300)
    right_arm.set_gripper_position(int(np.clip(right_grip * 10.0, 0, 840)), wait=False, speed=300)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Adam deployment client — connects serve_wam.py to the real arms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Inference server
    p.add_argument("--policy-host", default="localhost", help="serve_wam.py host")
    p.add_argument("--policy-port", type=int, default=5000, help="serve_wam.py port")

    # ZMQ cameras (pub_zed.py on the Jetson)
    p.add_argument("--zmq-host", default="192.222.10.10", help="Jetson IP for ZMQ camera feeds")
    p.add_argument("--head-port", type=int, default=5566, help="ZMQ port for head camera")
    p.add_argument("--left-wrist-port", type=int, default=5568, help="ZMQ port for left wrist camera")
    p.add_argument("--right-wrist-port", type=int, default=5569, help="ZMQ port for right wrist camera")

    # xArm IPs
    p.add_argument("--left-arm-ip", default="192.168.10.22", help="Left xArm 6 IP")
    p.add_argument("--right-arm-ip", default="192.168.10.201", help="Right xArm 6 IP")

    # Task
    p.add_argument("--prompt", default="pick up the object", help="Language instruction for the policy")

    # Control
    p.add_argument("--inference-freq", type=float, default=10.0, help="Control loop frequency (Hz)")
    p.add_argument("--duration-s", type=float, default=600.0, help="Total run duration (seconds)")
    p.add_argument("--chunk-size", type=int, default=24, help="Action horizon (must match model)")
    p.add_argument("--smooth-window", type=int, default=5,
                   help="Savitzky-Golay window (steps) for smoothing chunk boundaries; 0 to disable")

    # Safety
    p.add_argument("--max-joint-jump-deg", type=float, default=30.0,
                   help="Skip action if any joint would move more than this many degrees")
    p.add_argument("--joint-speed-deg-s", type=float, default=30.0,
                   help="xArm max joint speed (deg/s)")
    p.add_argument("--joint-acc-deg-s2", type=float, default=200.0,
                   help="xArm max joint acceleration (deg/s²)")
    p.add_argument("--frame-max-age-s", type=float, default=0.5,
                   help="Max camera frame age before skipping inference")

    # Motion enable
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True,
                   help="Log targets without moving. Use --no-dry-run to enable motion.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    speed_rad_s = float(np.radians(args.joint_speed_deg_s))
    acc_rad_s2  = float(np.radians(args.joint_acc_deg_s2))
    max_jump_rad = float(np.radians(args.max_joint_jump_deg))

    logger.info("=" * 70)
    logger.info("Adam deployment client")
    logger.info("  server:   ws://%s:%d", args.policy_host, args.policy_port)
    logger.info("  cameras:  zmq://%s  head=%d left=%d right=%d",
                args.zmq_host, args.head_port, args.left_wrist_port, args.right_wrist_port)
    logger.info("  arms:     left=%s  right=%s", args.left_arm_ip, args.right_arm_ip)
    logger.info("  rate:     %.1f Hz  duration=%.0fs  dry_run=%s",
                args.inference_freq, args.duration_s, args.dry_run)
    logger.info("  prompt:   %r", args.prompt)
    logger.info("=" * 70)

    # ── Cameras ───────────────────────────────────────────────────────────
    cams = {
        "head_left":   _ZmqCamSubscriber("head_left",   args.zmq_host, args.head_port),
        "left_wrist":  _ZmqCamSubscriber("left_wrist",  args.zmq_host, args.left_wrist_port),
        "right_wrist": _ZmqCamSubscriber("right_wrist", args.zmq_host, args.right_wrist_port),
    }
    for cam in cams.values():
        cam.start()

    logger.info("Waiting for first frame from all cameras (10s timeout)...")
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if all(cam.get_latest(1.0) is not None for cam in cams.values()):
            break
        time.sleep(0.1)
    missing = [n for n, c in cams.items() if c.get_latest(1.0) is None]
    if missing:
        raise RuntimeError(f"Timed out waiting for cameras: {missing}")
    logger.info("All camera feeds active.")

    # ── Arms ──────────────────────────────────────────────────────────────
    logger.info("Connecting to arms...")
    left_arm  = XArmAPI(args.left_arm_ip)
    right_arm = XArmAPI(args.right_arm_ip)
    _prepare_arm(left_arm, "left")
    _prepare_arm(right_arm, "right")

    # Current position becomes the safe home (no auto-homing to a fixed pose).
    left_home_joints, left_home_grip   = _read_arm_state(left_arm, "left")
    right_home_joints, right_home_grip = _read_arm_state(right_arm, "right")
    logger.info("Home position captured (current pose).")
    logger.info("  left  joints (rad): %s  grip=%.2f", left_home_joints, left_home_grip)
    logger.info("  right joints (rad): %s  grip=%.2f", right_home_joints, right_home_grip)

    # ── Policy server ─────────────────────────────────────────────────────
    logger.info("Connecting to policy server...")
    policy = WamClientPolicy(host=args.policy_host, port=args.policy_port)
    broker = AsyncActionChunkBroker(policy=policy, action_horizon=args.chunk_size, smooth_window=args.smooth_window)
    logger.info("Policy connected (chunk_size=%d, smooth_window=%d, async).", args.chunk_size, args.smooth_window)

    # ── Safety gate for live motion ───────────────────────────────────────
    if not args.dry_run:
        logger.warning("!!! LIVE MOTION ENABLED. Press Enter to start, Ctrl-C to abort.")
        try:
            input()
        except EOFError:
            pass

    # ── Control loop ──────────────────────────────────────────────────────
    period = 1.0 / args.inference_freq
    start_t = time.monotonic()
    next_tick = start_t
    iteration = 0

    try:
        while time.monotonic() - start_t < args.duration_s:
            iteration += 1

            # Read cameras and arm state for this control step.
            head = cams["head_left"].get_latest(args.frame_max_age_s)
            lw   = cams["left_wrist"].get_latest(args.frame_max_age_s)
            rw   = cams["right_wrist"].get_latest(args.frame_max_age_s)
            if head is None or lw is None or rw is None:
                logger.warning("Iter %d: stale/missing frames — skipping", iteration)
                next_tick += period
                time.sleep(max(0.0, next_tick - time.monotonic()))
                continue

            left_joints, left_grip   = _read_arm_state(left_arm, "left")
            right_joints, right_grip = _read_arm_state(right_arm, "right")

            # obs_fn is called by the broker at chunk boundaries to capture a
            # fresh observation for the next inference — always reflects real state.
            def obs_fn(_h=head, _l=lw, _r=rw):
                h = cams["head_left"].get_latest(args.frame_max_age_s)
                l = cams["left_wrist"].get_latest(args.frame_max_age_s)
                r = cams["right_wrist"].get_latest(args.frame_max_age_s)
                h = _h if h is None else h
                l = _l if l is None else l
                r = _r if r is None else r
                lj, lg = _read_arm_state(left_arm, "left")
                rj, rg = _read_arm_state(right_arm, "right")
                return {
                    "observation/head_left":   h,
                    "observation/left_wrist":  l,
                    "observation/right_wrist": r,
                    "observation/state": np.array(lj + [lg] + rj + [rg], dtype=np.float32),
                    "prompt": args.prompt,
                }

            t0 = time.monotonic()
            action = broker.get_action(obs_fn)   # (14,) float32
            infer_ms = (time.monotonic() - t0) * 1000.0

            # Safety: skip if any joint would jump too far
            left_target  = action[0:6]
            right_target = action[7:13]
            left_jump_rad  = float(np.max(np.abs(left_target  - np.array(left_joints))))
            right_jump_rad = float(np.max(np.abs(right_target - np.array(right_joints))))

            if max(left_jump_rad, right_jump_rad) > max_jump_rad:
                logger.warning(
                    "Iter %d SKIP: jump L=%.1f° R=%.1f° exceeds limit %.1f°",
                    iteration,
                    np.degrees(left_jump_rad),
                    np.degrees(right_jump_rad),
                    args.max_joint_jump_deg,
                )
            elif args.dry_run:
                if iteration % 10 == 1:
                    logger.info(
                        "Iter %d DRY: infer=%.0fms jump L=%.1f° R=%.1f° "
                        "gripL %.2f→%.2f gripR %.2f→%.2f",
                        iteration, infer_ms,
                        np.degrees(left_jump_rad), np.degrees(right_jump_rad),
                        left_grip, float(action[6]),
                        right_grip, float(action[13]),
                    )
            else:
                _apply_arm_action(left_arm, right_arm, action, speed_rad_s, acc_rad_s2)
                if iteration % 10 == 1:
                    logger.info(
                        "Iter %d LIVE: infer=%.0fms jump L=%.1f° R=%.1f°",
                        iteration, infer_ms,
                        np.degrees(left_jump_rad), np.degrees(right_jump_rad),
                    )

            # Rate limiting
            next_tick += period
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            elif sleep_s < -3 * period:
                logger.warning("Falling behind by %.2fs; resetting tick", -sleep_s)
                next_tick = time.monotonic()

    except KeyboardInterrupt:
        logger.info("Interrupted — stopping.")
    finally:
        logger.info("Cleaning up...")
        for cam in cams.values():
            cam.stop()
        for arm, label in [(left_arm, "left"), (right_arm, "right")]:
            try:
                arm.set_state(state=4)   # stop
                arm.disconnect()
                logger.info("%s arm disconnected", label)
            except Exception:
                logger.exception("%s arm cleanup failed", label)
        policy.close()


if __name__ == "__main__":
    main()
