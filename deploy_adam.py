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

# Gripper actuation speed (xArm units). The standalone test confirmed full open<->close
# works at 2000 with wait=True; at the old 300 + wait=False it crawled and looked stuck.
GRIPPER_SPEED = 3000

# Gripper close threshold (raw/10 units). Below this -> fully closed (0); at/above -> fully
# open (85). Binarizes the model's soft/partial gripper output into a decisive open/close,
# mirroring upstream YAM's DreamZeroJointPosClient (which snaps the gripper at 0.5).
GRIPPER_CLOSE_THRESH = 30.0
# Default: BINARY gripper (open/close) at thresh 30. The q99 cap limits the model's per-chunk
# open to ~37 raw/10, so the threshold MUST stay below ~37 (30 leaves margin) or a release from
# closed never triggers "open". Use --no-gripper-binary for continuous pass-through.
BINARIZE_GRIP = True

# Gripper rounding step (raw/10 units). >0 quantizes the continuous prediction by rounding DOWN
# to a multiple (e.g. step=10: 32->30, 26->20, 84->80), snapping out sub-step jitter while keeping
# a graded gripper. Only applies in continuous mode (BINARIZE_GRIP off already snaps to 0/85).
# NOTE: floor removes sub-step noise only; a value oscillating around a bucket edge (e.g. ~30)
# still flips 20<->30 step-to-step -> use hysteresis for boundary chatter, not rounding.
GRIPPER_ROUND_STEP = 0.0


def _binarize_grip(g: float) -> float:
    """Binary gripper (raw/10): < GRIPPER_CLOSE_THRESH -> 0 (closed), else -> 85 (open/max).
    Mirrors upstream YAM's DreamZeroJointPosClient (snaps the gripper at 0.5 on its [0,1] scale)."""
    return 0.0 if g < GRIPPER_CLOSE_THRESH else 85.0


def _round_grip(g: float, step: float) -> float:
    """Quantize the continuous gripper (raw/10) by rounding DOWN to a multiple of step (e.g.
    step=10: 32->30, 26->20, 84->80). Floor (not nearest) so the gripper biases toward CLOSE
    (smaller raw/10 = more closed) -> firmer grasps. Snaps out sub-step jitter, graded gripper."""
    return float(np.floor(g / step) * step)


def _grip_command(g: float) -> float:
    """Map the model's raw/10 gripper prediction to the value sent to the arm. Dispatches to the
    active gripper option, in precedence order:
      1. BINARIZE_GRIP          -> _binarize_grip: decisive 0/85 (grasp). DEFAULT.
      2. GRIPPER_ROUND_STEP > 0 -> _round_grip:    graded, quantized to the step.
      3. else                   -> continuous pass-through (matches training)."""
    if BINARIZE_GRIP:
        return _binarize_grip(g)
    if GRIPPER_ROUND_STEP > 0:
        return _round_grip(g, GRIPPER_ROUND_STEP)
    return g


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

def _prepare_arm(arm: XArmAPI, label: str,
                 tcp_payload_kg: float = 0.0, tcp_cog_mm=(0.0, 0.0, 0.0)) -> None:
    arm.motion_enable(enable=True)
    arm.clean_warn()
    arm.clean_error()
    time.sleep(0.2)
    # Gravity compensation: tell the controller the gripper(+payload) mass so the arm holds
    # its pose instead of sagging (sag drifts joint torque -> force-guard false trips).
    if tcp_payload_kg > 0:
        code = arm.set_tcp_load(tcp_payload_kg, list(tcp_cog_mm))
        logger.info("%s set_tcp_load(%.3f kg, cog=%s mm) -> code=%d",
                    label, tcp_payload_kg, list(tcp_cog_mm), code)
        time.sleep(0.1)
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


def _read_joint_torque(arm: XArmAPI) -> Optional[np.ndarray]:
    """Per-joint torque (Nm, 6,) estimated from motor current, or None on read error.
    Includes gravity + payload, so use it as an EXCESS-over-baseline signal (contact spike),
    not an absolute force. No F/T sensor required (uses get_joints_torque)."""
    code, tq = arm.get_joints_torque()
    if code != 0 or not tq:
        return None
    return np.asarray(tq[:6], dtype=np.float64)


def _apply_arm_action(
    left_arm: XArmAPI,
    right_arm: XArmAPI,
    action: np.ndarray,
    speed_rad_s: float,
    acc_rad_s2: float,
) -> None:
    """Send 14-D action to both arms (non-blocking)."""
    left_joints  = action[0:6].tolist()
    left_grip    = _grip_command(float(action[6]))    # gripper: binary by default; --no-gripper-binary [+ --gripper-round-step]
    right_joints = action[7:13].tolist()
    right_grip   = _grip_command(float(action[13]))

    left_arm.set_servo_angle(
        angle=left_joints, is_radian=True, wait=False,
        speed=speed_rad_s, mvacc=acc_rad_s2,
    )
    right_arm.set_servo_angle(
        angle=right_joints, is_radian=True, wait=False,
        speed=speed_rad_s, mvacc=acc_rad_s2,
    )
    # Gripper: model outputs raw/10 (0-84); set_gripper_position takes 0-850 (850=open, 0=closed),
    # so ×10 is required. High speed + wait=False so it actuates between control steps instead of
    # crawling (speed=300 was too slow to track a moving target -> looked like it never moved).
    left_arm.set_gripper_position(int(np.clip(left_grip * 10.0, 0, 850)), wait=False, speed=GRIPPER_SPEED)
    right_arm.set_gripper_position(int(np.clip(right_grip * 10.0, 0, 850)), wait=False, speed=GRIPPER_SPEED)


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
    p.add_argument("--inference-freq", type=float, default=30.0,
                   help="Control loop / open-loop playback frequency (Hz). Default 30 = the dataset "
                        "fps, so the chunk plays at the speed it was trained.")
    p.add_argument("--duration-s", type=float, default=600.0, help="Total run duration (seconds)")
    p.add_argument("--chunk-size", type=int, default=24, help="Action horizon (must match model)")
    p.add_argument("--smooth-window", type=int, default=5,
                   help="Savitzky-Golay window (steps) for smoothing chunk boundaries; 0 to disable")

    # Safety
    p.add_argument("--max-joint-jump-deg", type=float, default=30.0,
                   help="Skip action if any joint would move more than this many degrees")
    p.add_argument("--joint-speed-deg-s", type=float, default=80.0,
                   help="xArm max joint speed (deg/s)")
    p.add_argument("--joint-acc-deg-s2", type=float, default=400.0,
                   help="xArm max joint acceleration (deg/s²)")
    p.add_argument("--frame-max-age-s", type=float, default=0.5,
                   help="Max camera frame age before skipping inference")

    # Motion enable
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True,
                   help="Log targets without moving. Use --no-dry-run to enable motion.")
    # Control loop — default is YAM-aligned (synchronous receding horizon from fresh obs).
    p.add_argument("--open-loop-horizon", type=int, default=8,
                   help="Execute this many actions of each predicted chunk before replanning "
                        "from a fresh observation (upstream YAM uses 8). Must be <= --chunk-size.")
    p.add_argument("--async-prefetch", action="store_true",
                   help="Use the async pre-fetch broker (executes the full chunk, no inference "
                        "holds -> smooth). Pair with --reanchor (default on) to avoid the "
                        "stale-anchor snap-back oscillation.")
    p.add_argument("--reanchor", action=argparse.BooleanOptionalAction, default=False,
                   help="Async only: re-base each pre-fetched chunk's joint targets to the LIVE "
                        "pose at activation (cmd = chunk - chunk[0] + current_state). Removes the "
                        "snap-back oscillation while keeping async smoothness. Default OFF; "
                        "--reanchor to enable.")
    p.add_argument("--reanchor-skip", type=int, default=2,
                   help="Async + --reanchor: drop this many leading steps of each chunk and anchor "
                        "AT that index, so the first executed command == the live pose. Higher = "
                        "reanchors 'harder' (skips more of the stale snap-back start). Default 2.")
    p.add_argument("--chunk-tail-skip", type=int, default=0,
                   help="Async: stop each chunk this many steps BEFORE the end (drop the trailing, "
                        "most-extrapolated steps) and swap to the fresher pre-fetched chunk earlier. "
                        "Default 0.")

    # Force guard (joint-torque based; no F/T sensor required)
    p.add_argument("--force-stop", action=argparse.BooleanOptionalAction, default=False,
                   help="Monitor joint-torque EXCESS over the free-motion baseline; if it exceeds "
                        "HALF of --estop-torque (i.e. 2x more sensitive than the e-stop), stop the "
                        "current chunk and regenerate from the contact pose. CALIBRATE --estop-torque first.")
    p.add_argument("--estop-torque", type=float, default=20.0,
                   help="E-stop boundary: joint-torque EXCESS (Nm) over baseline at which the hard "
                        "collision/e-stop trips. The soft guard fires at HALF this value. "
                        "PLACEHOLDER default 20 Nm -- MUST be calibrated to YOUR arm + payload.")
    p.add_argument("--force-max-trips", type=int, default=5,
                   help="Consecutive force trips before a HARD stop (set_state 4), to avoid pressing "
                        "in a regenerate loop. Default 5.")
    p.add_argument("--tcp-payload-kg", type=float, default=0.0,
                   help="Payload mass (kg) at the TCP (gripper + held object) for gravity "
                        "compensation via set_tcp_load. 0 = don't set (current behavior). Set this "
                        "if an arm sags / its torque drifts (causes force-guard false trips).")
    p.add_argument("--tcp-cog-mm", type=str, default="0,0,0",
                   help="TCP center of gravity 'x,y,z' in mm for set_tcp_load. Default 0,0,0.")
    p.add_argument("--gripper-close-thresh", type=float, default=GRIPPER_CLOSE_THRESH,
                   help="Binary gripper threshold (raw/10): predicted gripper below this -> fully "
                        "closed (grasp), at/above -> fully open. Keep BELOW ~37 (the q99 open cap); "
                        "default 30. Too high and releases from closed never fire.")
    p.add_argument("--gripper-binary", action=argparse.BooleanOptionalAction, default=True,
                   help="Binarize the gripper into open/close at --gripper-close-thresh. Default OFF: "
                        "send the raw continuous predicted gripper value (matches training).")
    p.add_argument("--gripper-round-step", type=float, default=0.0,
                   help="Continuous mode only (--no-gripper-binary): round the predicted gripper "
                        "(raw/10) DOWN to a multiple of this step, e.g. 10 -> 32->30, 26->20, "
                        "84->80. Floor biases toward close (firmer grasp). Snaps out sub-step jitter "
                        "while keeping a graded gripper. 0 = off. Ignored when --gripper-binary is on.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    global GRIPPER_CLOSE_THRESH, BINARIZE_GRIP, GRIPPER_ROUND_STEP
    GRIPPER_CLOSE_THRESH = args.gripper_close_thresh
    BINARIZE_GRIP = args.gripper_binary  # binary is the DEFAULT (--no-gripper-binary for continuous)
    GRIPPER_ROUND_STEP = max(0.0, args.gripper_round_step)
    if BINARIZE_GRIP:
        grip_desc = f"BINARY (open/close at thresh {GRIPPER_CLOSE_THRESH:.0f} raw/10)"
    elif GRIPPER_ROUND_STEP > 0:
        grip_desc = f"continuous, rounded DOWN to multiple of {GRIPPER_ROUND_STEP:.0f} (raw/10)"
    else:
        grip_desc = "continuous (q99 prediction passed through)"
    logger.info("Gripper: %s", grip_desc)
    if BINARIZE_GRIP and GRIPPER_ROUND_STEP > 0:
        logger.warning("--gripper-round-step=%.0f ignored: --gripper-binary is on "
                       "(binary already snaps to 0/85). Use --no-gripper-binary to round.",
                       GRIPPER_ROUND_STEP)
    if BINARIZE_GRIP and GRIPPER_CLOSE_THRESH > 37:
        logger.warning("gripper-close-thresh=%.0f > ~37 (q99 open cap): releases from closed may "
                       "NEVER trigger 'open'. Consider --gripper-close-thresh 25-30.",
                       GRIPPER_CLOSE_THRESH)
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
    tcp_cog = tuple(float(x) for x in args.tcp_cog_mm.split(","))
    _prepare_arm(left_arm, "left", args.tcp_payload_kg, tcp_cog)
    _prepare_arm(right_arm, "right", args.tcp_payload_kg, tcp_cog)

    # Current position becomes the safe home (no auto-homing to a fixed pose).
    left_home_joints, left_home_grip   = _read_arm_state(left_arm, "left")
    right_home_joints, right_home_grip = _read_arm_state(right_arm, "right")
    logger.info("Home position captured (current pose).")
    logger.info("  left  joints (rad): %s  grip=%.2f", left_home_joints, left_home_grip)
    logger.info("  right joints (rad): %s  grip=%.2f", right_home_joints, right_home_grip)

    # ── Policy server ─────────────────────────────────────────────────────
    logger.info("Connecting to policy server...")
    policy = WamClientPolicy(host=args.policy_host, port=args.policy_port)
    # Default = YAM-aligned control loop: synchronous receding horizon, replan from the
    # current observation every --open-loop-horizon steps (matches upstream YAM's
    # DreamZeroJointPosClient). --async-prefetch opts back into the legacy async broker.
    sync = not args.async_prefetch
    broker = AsyncActionChunkBroker(policy=policy, action_horizon=args.chunk_size,
                                    smooth_window=args.smooth_window, sync=sync,
                                    open_loop_horizon=args.open_loop_horizon,
                                    reanchor=args.reanchor, reanchor_skip=args.reanchor_skip,
                                    chunk_tail_skip=args.chunk_tail_skip)
    logger.info("Policy connected (chunk_size=%d, open_loop_horizon=%d, mode=%s%s).",
                args.chunk_size, args.open_loop_horizon,
                "async-prefetch" if args.async_prefetch else "sync receding-horizon (YAM-aligned)",
                f", reanchor={args.reanchor}" if args.async_prefetch else "")

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
    tq_base = {"left": None, "right": None}   # per-arm joint-torque EMA baseline (free motion)
    force_trips = 0
    force_quiet = False   # once a trip fires, stop the per-iteration [force] spam

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

            # ── Force guard: stop current chunk & regenerate if pressing too hard ──
            # Soft trip = HALF the e-stop boundary (2x more sensitive), measured as joint-torque
            # excess over a slow free-motion baseline (gravity/payload live in the baseline).
            if args.force_stop:
                soft = args.estop_torque / 2.0
                excess = 0.0
                worst_arm, worst_joint = "?", 0   # which arm/joint drives the max excess
                dbg = []
                for lab, a in (("left", left_arm), ("right", right_arm)):
                    cur = _read_joint_torque(a)
                    if cur is None:
                        dbg.append(f"{lab}: TORQUE-READ-FAILED")
                        continue
                    base = tq_base[lab]
                    if base is None:
                        tq_base[lab] = cur
                        dbg.append(f"{lab}: baseline-init tau={np.round(cur, 1).tolist()}")
                        continue
                    diff = np.abs(cur - base)
                    e = float(np.max(diff))
                    j = int(np.argmax(diff))          # which joint drives the excess
                    if e > excess:
                        excess, worst_arm, worst_joint = e, lab, j
                    if e < soft:                      # track slow drift (gravity sag) up to the trip
                        tq_base[lab] = 0.95 * base + 0.05 * cur   # only a FAST spike outruns this -> trips
                    dbg.append(f"{lab}: tau={np.round(cur, 1).tolist()} "
                               f"base={np.round(base, 1).tolist()} exc={e:.1f}@J{j + 1}")
                # Force debug: print every ~10 steps, and go QUIET once a trip/e-stop has fired so
                # the actual stopping force (the TRIP / HARD STOP lines) isn't buried in the stream.
                if not force_quiet and iteration % 10 == 1:
                    logger.info("[force] iter=%d excess=%.1f soft=%.1f estop=%.1f trips=%d | %s",
                                iteration, excess, soft, args.estop_torque, force_trips, "  ".join(dbg))
                if excess > soft:
                    force_trips += 1
                    force_quiet = True   # suppress the per-iteration line from here on
                    logger.warning("Iter %d FORCE TRIP: %s arm J%d excess %.1f Nm > soft %.1f "
                                   "(e-stop %.1f) -> hold + regenerate (#%d/%d)",
                                   iteration, worst_arm.upper(), worst_joint + 1, excess, soft,
                                   args.estop_torque, force_trips, args.force_max_trips)
                    if not args.dry_run:
                        # Stop driving into contact: command the CURRENT measured pose (hold).
                        lj, _ = _read_arm_state(left_arm, "left")
                        rj, _ = _read_arm_state(right_arm, "right")
                        left_arm.set_servo_angle(angle=lj, is_radian=True, wait=False,
                                                 speed=speed_rad_s, mvacc=acc_rad_s2)
                        right_arm.set_servo_angle(angle=rj, is_radian=True, wait=False,
                                                  speed=speed_rad_s, mvacc=acc_rad_s2)
                    broker.force_replan()        # abandon this chunk; fresh inference next tick
                    if force_trips >= args.force_max_trips:
                        logger.error("Iter %d: %d consecutive force trips -> HARD STOP "
                                     "(raise --estop-torque / --force-max-trips if false-tripping).",
                                     iteration, force_trips)
                        if not args.dry_run:
                            left_arm.set_state(4)
                            right_arm.set_state(4)
                        break
                    next_tick += period
                    time.sleep(max(0.0, next_tick - time.monotonic()))
                    continue
                else:
                    force_trips = 0

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
                if iteration % 5 == 1:
                    # gripper: act = physical position now (raw, 0-850); pred = model output (raw/10);
                    # cmd = value sent to SDK. If act doesn't follow cmd -> hardware; if pred stays
                    # high -> action model never commands a close (disagrees with the video model).
                    _, la_now = left_arm.get_gripper_position()
                    _, ra_now = right_arm.get_gripper_position()
                    logger.info(
                        "Iter %d LIVE: infer=%.0fms jump L=%.1f° R=%.1f° | "
                        "gripL act=%s pred=%.1f cmd=%d  gripR act=%s pred=%.1f cmd=%d",
                        iteration, infer_ms,
                        np.degrees(left_jump_rad), np.degrees(right_jump_rad),
                        la_now, float(action[6]),  int(_grip_command(float(action[6]))  * 10.0),
                        ra_now, float(action[13]), int(_grip_command(float(action[13])) * 10.0),
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
