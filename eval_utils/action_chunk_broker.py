"""Async action chunk broker matching DreamZero's real-robot deployment strategy.

Key principles (from DreamZero paper §4 and YAM deployment):
  1. Execute the FULL action chunk open-loop — chunk is the atomic unit of execution.
  2. Start background inference at step 0 of each new chunk (not mid-chunk) so the
     next chunk is ready as soon as the current one finishes.
  3. At chunk boundary: swap immediately if ready, otherwise hold the last action
     (arm stays put) until inference completes — avoids executing stale actions.
  4. Savitzky-Golay smoothing blends the seam between consecutive chunks to
     prevent jerky transitions when the new chunk arrives.
  5. Fresh camera + state observation is captured at the chunk boundary and used
     for the next inference call, not the stale obs from step 0.

With 24-step chunks at 10 Hz (2.4s per chunk) and ~3s inference on H100:
  - Inference starts at step 0 → runs for 2.4s while chunk executes
  - 0.6s wait at boundary → next chunk starts
  - Total effective policy rate: ~3s per chunk ≈ 0.33 Hz policy updates
"""

import logging
import threading
from typing import Callable

import numpy as np
from scipy.signal import savgol_filter

from eval_utils.wam_client import WamClientPolicy

logger = logging.getLogger(__name__)


def _smooth_boundary(prev_chunk: np.ndarray, next_chunk: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply Savitzky-Golay smoothing at the seam between two consecutive chunks.

    Blends the tail of prev_chunk into the head of next_chunk so joint
    velocity is continuous at the transition.

    Args:
        prev_chunk: (T, D) — the chunk that just finished executing.
        next_chunk: (T, D) — the incoming chunk.
        window:     number of steps on each side to use for blending.

    Returns:
        Smoothed next_chunk (T, D).
    """
    if prev_chunk is None or window <= 0:
        return next_chunk

    T, D = next_chunk.shape
    window = min(window, T, len(prev_chunk))
    if window < 3:
        return next_chunk

    # Concatenate tail of previous chunk with head of next chunk and smooth.
    tail = prev_chunk[-window:]          # (window, D)
    head = next_chunk[:window]          # (window, D)
    concat = np.concatenate([tail, head], axis=0)  # (2*window, D)

    poly = min(3, 2 * window - 1)
    win_len = 2 * window if (2 * window) % 2 == 1 else 2 * window - 1
    smoothed = savgol_filter(concat, window_length=win_len, polyorder=poly, axis=0)

    result = next_chunk.copy()
    result[:window] = smoothed[window:]
    return result


class AsyncActionChunkBroker:
    """Non-blocking action chunk broker — DreamZero deployment strategy.

    Usage:
        broker = AsyncActionChunkBroker(policy, action_horizon=24)
        obs_fn = lambda: {...}   # callable that returns fresh obs dict
        for step in range(max_steps):
            action = broker.get_action(obs_fn)   # (14,) float32, never blocks after cold start
            env.apply_action(action)
    """

    def __init__(
        self,
        policy: WamClientPolicy,
        action_horizon: int = 24,
        smooth_window: int = 5,
        sync: bool = False,
        open_loop_horizon: int | None = None,
    ) -> None:
        self._policy = policy
        self._action_horizon = action_horizon
        self._smooth_window = smooth_window
        # Receding horizon: execute this many actions of each predicted chunk before
        # replanning. Upstream YAM (DreamZeroJointPosClient) uses 8. Defaults to the
        # full chunk (legacy behavior) when not set.
        self._open_loop_horizon = open_loop_horizon if open_loop_horizon else action_horizon
        # sync=True: replan-at-completion. Execute the full chunk, THEN capture a FRESH
        # observation and (blockingly) regenerate. Each chunk is anchored to the live
        # state at generation time -> no one-chunk-stale anchor, no snap-back oscillation.
        # Cost: a brief hold (~inference time) at each chunk boundary (arm holds last pose).
        # sync=False (default): async pre-fetch (next inference starts at chunk start using
        # obs from ~one chunk ago -> can oscillate on dynamic tasks).
        self._sync = sync

        self._chunk: np.ndarray | None = None       # current chunk executing
        self._prev_chunk: np.ndarray | None = None  # finished chunk (for smoothing)
        self._step = 0

        self._next_chunk: np.ndarray | None = None  # pre-fetched next chunk
        self._next_lock = threading.Lock()
        self._inflight = threading.Event()

    def get_action(self, obs_fn: Callable[[], dict]) -> np.ndarray:
        """Return next (14,) action.

        Args:
            obs_fn: Zero-argument callable that returns a fresh observation dict.
                    Called only at chunk boundaries to feed the next inference.
                    Calling it every step is fine — it's cheap.
        """
        # ── Sync mode: YAM-style receding horizon ───────────────────────────
        # Mirrors upstream DreamZeroJointPosClient.infer: replan from a FRESH observation
        # every open_loop_horizon steps (blocking), then execute chunk[step]. Each chunk is
        # anchored to the live state at generation time -> no stale anchor, no snap-back.
        # No seam smoothing (YAM doesn't): the fresh-obs anchor makes chunk[0] ~= current pose.
        if self._sync:
            if self._chunk is None or self._step >= self._open_loop_horizon:
                obs = obs_fn()                       # fresh obs NOW (live state anchor)
                self._chunk = self._fetch(obs)       # blocking inference; arm holds last pose
                self._step = 0
                logger.debug("Sync replan from fresh obs (receding horizon=%d).", self._open_loop_horizon)
            return self._advance()

        # ── Cold start ──────────────────────────────────────────────────────
        if self._chunk is None:
            logger.info("Cold-start inference (blocking)...")
            obs = obs_fn()
            self._chunk = self._fetch(obs)
            self._step = 0
            # Immediately kick off background inference for chunk 1.
            self._start_background_fetch(obs_fn)
            return self._advance()

        # ── Chunk exhausted: try to swap ────────────────────────────────────
        if self._step >= self._action_horizon:
            with self._next_lock:
                if self._next_chunk is not None:
                    self._prev_chunk = self._chunk
                    self._chunk = _smooth_boundary(
                        self._prev_chunk, self._next_chunk, self._smooth_window
                    )
                    self._next_chunk = None
                    self._step = 0
                    logger.debug("Swapped to next chunk (smoothed).")
                    # Start inference for chunk N+1 immediately.
                    self._start_background_fetch(obs_fn)
                else:
                    # Inference not done yet — hold last action (arm stays put).
                    logger.debug("Next chunk not ready; holding last action.")
                    return self._chunk[-1].copy()

        return self._advance()

    def _advance(self) -> np.ndarray:
        action = self._chunk[self._step].copy()
        self._step += 1
        return action

    def _fetch(self, obs: dict) -> np.ndarray:
        result = self._policy.infer(obs)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]
        # Debug: does the action model EVER intend to close the gripper within the chunk?
        # If L/R min stays high (~open) across all steps, the action head is not commanding a
        # grasp (disagrees with the video model). gripper dims: 6=L, 13=R (raw/10, ~84 open, ~28 closed).
        if actions.shape[-1] >= 14:
            lg, rg = actions[:, 6], actions[:, 13]
            logger.info("grip CHUNK (raw/10): L[min=%.0f max=%.0f first=%.0f last=%.0f] "
                        "R[min=%.0f max=%.0f first=%.0f last=%.0f]",
                        lg.min(), lg.max(), lg[0], lg[-1], rg.min(), rg.max(), rg[0], rg[-1])
        return actions  # (T, 14)

    def _start_background_fetch(self, obs_fn: Callable[[], dict]) -> None:
        if self._inflight.is_set():
            return
        self._inflight.set()
        t = threading.Thread(
            target=self._background_fetch, args=(obs_fn,), daemon=True, name="infer_bg"
        )
        t.start()

    def _background_fetch(self, obs_fn: Callable[[], dict]) -> None:
        try:
            obs = obs_fn()   # capture fresh observation now, at inference start
            chunk = self._fetch(obs)
            with self._next_lock:
                self._next_chunk = chunk
            logger.debug("Background inference complete.")
        except Exception:
            logger.exception("Background inference failed.")
        finally:
            self._inflight.clear()

    def reset(self) -> None:
        with self._next_lock:
            self._chunk = None
            self._prev_chunk = None
            self._next_chunk = None
            self._step = 0
