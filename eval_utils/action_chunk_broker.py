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
import time
from typing import Callable

import numpy as np
from scipy.signal import savgol_filter

from eval_utils.wam_client import WamClientPolicy

logger = logging.getLogger(__name__)

# Joint dims of the 14-D action (L arm 0-5, R arm 7-12). Grippers (6, 13) are excluded
# from re-anchoring — they're near-binary and barely drift over a single chunk.
_JOINT_IDX = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])


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
        reanchor: bool = True,
        reanchor_skip: int = 2,
        chunk_tail_skip: int = 0,
        rtc: bool = False,
        inference_freq: float | None = None,
    ) -> None:
        self._policy = policy
        self._action_horizon = action_horizon
        self._smooth_window = smooth_window
        # Re-anchoring (async only): the model's actions are relative, reconstructed by the
        # server to absolute against the observation's joint state. In async pre-fetch that
        # observation is ~one inference (~3s) stale, so the chunk's absolute targets are anchored
        # to where the arm WAS, which snaps it back -> oscillation (worst near the object, where
        # corrective moves are small). Re-anchoring re-bases each new chunk's joint targets so
        # step 0 == the LIVE pose at activation: cmd[k] = chunk[k] - chunk[0] + live_state.
        # Keeps async's hold-free smoothness while removing the stale-anchor snap-back.
        # Grippers (dims 6,13) are left as server-returned (they barely drift over one chunk).
        self._reanchor = reanchor
        # async only: drop this many leading steps of each chunk and anchor AT that index, so the
        # first executed command == the LIVE pose -> kills the stale-obs snap-back ("going back to
        # where the arm was"). Higher = reanchors harder (skips more of the stale chunk start).
        self._reanchor_skip = max(0, int(reanchor_skip))
        # async only: stop executing each chunk this many steps BEFORE the end (drop the trailing,
        # most-extrapolated steps) and swap to the fresher pre-fetched chunk earlier.
        self._chunk_tail_skip = max(0, min(int(chunk_tail_skip), action_horizon - 1))
        self._anchor_base: np.ndarray | None = None   # chunk[reanchor_skip] of the active chunk
        self._anchor_state: np.ndarray | None = None  # live 14-D joint state at activation
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
        self._gen = 0   # bumped by force_replan() to invalidate in-flight (stale-obs) inferences
        self._infer_lock = threading.Lock()   # serialize policy.infer(): one ws.recv() at a time
        # server-reported model forward-pass time (ms) of the most recent COMPLETED fetch — the
        # pure inference time per chunk (no deploy-side playback, no network wait). The inference
        # rate is 1/infer_s; logged per chunk in _fetch.
        self._last_infer_ms: float | None = None
        # bumped once per completed fetch (a fresh plan). The deploy client watches this to detect
        # a replan and time the execution (open-loop) phase between plans.
        self._fetch_count: int = 0

        # ── Real-Time Chunking (RTC, arXiv:2506.07339) ──────────────────────────────────────
        # Continuous async execution with prefix-inpainting: the next chunk is generated WHILE the
        # current one executes, conditioned on the committed leftover so it is continuous at the
        # seam (server does the inpainting). No reanchor / no Savitzky-Golay smoothing — RTC owns
        # continuity. Indices are tracked on an absolute timeline (_t) so a returned chunk can be
        # spliced at the position its observation was captured.
        self._rtc = rtc
        self._inference_freq = inference_freq
        self._t = 0                     # absolute control tick of the NEXT action to emit
        self._chunk_base_t = 0          # timeline tick that maps to self._chunk[0]
        self._next_base_t = 0           # timeline tick that maps to self._next_chunk[0]
        self._rtc_d = 1                 # measured inference delay in control steps (adapts)
        self._rtc_wall_ema: float | None = None   # EMA of obs->chunk wall-clock (seconds)

    @property
    def last_server_infer_ms(self) -> float | None:
        """Server-reported forward-pass time (ms) of the most recent fetch, or None. Lets the
        deploy client log the TRUE inference time instead of the broker round-trip (which is
        ~0 ms on the cached, non-replan ticks in sync mode)."""
        return self._last_infer_ms

    @property
    def fetch_count(self) -> int:
        """Number of completed fetches (fresh plans) so far. Increments on every inference;
        the deploy client diffs it to detect a replan and time the execution phase between plans."""
        return self._fetch_count

    def get_action(self, obs_fn: Callable[[], dict]) -> np.ndarray:
        """Return next (14,) action.

        Args:
            obs_fn: Zero-argument callable that returns a fresh observation dict.
                    Called only at chunk boundaries to feed the next inference.
                    Calling it every step is fine — it's cheap.
        """
        # ── RTC mode: continuous async with prefix-inpainting (owns its own path) ──
        if self._rtc:
            return self._get_action_rtc(obs_fn)

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
            self._set_anchor(obs)   # cold start: sent state == live state, so re-anchor is ~no-op
            self._step = self._start_step()   # skip stale leading steps when re-anchoring
            # Immediately kick off background inference for chunk 1.
            self._start_background_fetch(obs_fn)
            return self._advance()

        # ── Chunk exhausted: try to swap ────────────────────────────────────
        if self._step >= self._action_horizon - self._chunk_tail_skip:
            with self._next_lock:
                if self._next_chunk is not None:
                    self._prev_chunk = self._chunk
                    self._chunk = _smooth_boundary(
                        self._prev_chunk, self._next_chunk, self._smooth_window
                    )
                    self._next_chunk = None
                    self._step = self._start_step()   # skip stale leading steps when re-anchoring
                    # Re-anchor the incoming chunk to the LIVE pose now (it was generated from
                    # an obs ~one inference ago). One fresh obs read; reuse it for the next fetch.
                    obs = obs_fn()
                    self._set_anchor(obs)
                    logger.debug("Swapped to next chunk (smoothed, re-anchored).")
                    # Start inference for chunk N+1 immediately (reuse the obs we just read).
                    self._start_background_fetch(obs_fn, obs=obs)
                else:
                    # Inference not done yet — hold last action (arm stays put).
                    logger.debug("Next chunk not ready; holding last action.")
                    # Self-heal: make sure a prefetch is in flight (e.g. after a force_replan
                    # discarded the previous one) so we don't hold forever.
                    self._start_background_fetch(obs_fn)
                    return self._reanchor_action(self._chunk[-1].copy())

        return self._advance()

    def _set_anchor(self, obs: dict | None) -> None:
        """Record the active chunk's base (chunk[0]) and the live joint state, so the
        chunk's joint targets can be re-based to the current pose at execution time."""
        if not self._reanchor or self._sync or self._chunk is None:
            return
        idx = min(self._reanchor_skip, self._chunk.shape[0] - 1)
        self._anchor_base = self._chunk[idx].copy()
        state = obs.get("observation/state") if obs else None
        self._anchor_state = (
            np.asarray(state, dtype=np.float32) if state is not None else self._chunk[0].copy()
        )

    def _reanchor_action(self, action: np.ndarray) -> np.ndarray:
        """Re-base joint targets so chunk[0] maps to the live pose: cmd = a - base + live."""
        if (not self._reanchor or self._sync
                or self._anchor_base is None or self._anchor_state is None):
            return action
        action[_JOINT_IDX] = (
            action[_JOINT_IDX] - self._anchor_base[_JOINT_IDX] + self._anchor_state[_JOINT_IDX]
        )
        return action

    def _start_step(self) -> int:
        """First step index of a freshly-activated chunk. When re-anchoring (async), skip the
        stale leading steps so the first executed command == the live pose (no snap-back)."""
        if self._reanchor and not self._sync and self._chunk is not None:
            return min(self._reanchor_skip, self._chunk.shape[0] - 1)
        return 0

    def _advance(self) -> np.ndarray:
        action = self._chunk[self._step].copy()
        self._step += 1
        return self._reanchor_action(action)

    def _fetch(self, obs: dict) -> np.ndarray:
        # Single websocket -> only one infer (send+recv) at a time. Without this, a force_replan
        # cold-start fetch on the main thread races the in-flight background prefetch -> recv() on
        # the same socket from two threads -> ConcurrencyError. The main thread waits here for the
        # in-flight (stale) inference to finish; its result is then dropped by the _gen guard.
        with self._infer_lock:
            result = self._policy.infer(obs)
        self._fetch_count += 1
        # Server-reported model forward-pass time (ms) for THIS chunk. Logged per chunk as the
        # inference rate (1/infer_s) — pure model time, no deploy playback / network wait.
        st = result.get("server_timing") if isinstance(result, dict) else None
        if st is not None and "infer_ms" in st:
            self._last_infer_ms = float(st["infer_ms"])
            infer_s = self._last_infer_ms / 1000.0
            logger.info("[infer] %.2f Hz (model inference %.2fs / chunk)",
                        (1.0 / infer_s if infer_s > 0 else 0.0), infer_s)
            # Per-component breakdown (where the inference time goes), incl. each denoise step.
            m = st.get("model")
            if m:
                steps = "+".join(f"{x:.0f}" for x in m.get("diffusion_steps_ms", []))
                logger.info(
                    "[infer]   text_enc %.0fms  img_enc %.0fms  vae %.0fms  kv %.0fms  "
                    "diffusion %.0fms = [%s]ms (%d DIT steps)  sched %.0fms  total %.0fms",
                    m.get("text_encoder_ms", 0.0), m.get("image_encoder_ms", 0.0),
                    m.get("vae_ms", 0.0), m.get("kv_creation_ms", 0.0),
                    m.get("diffusion_ms", 0.0), steps, m.get("dit_compute_steps", 0),
                    m.get("scheduler_ms", 0.0), m.get("total_ms", 0.0),
                )
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

    def _start_background_fetch(self, obs_fn: Callable[[], dict], obs: dict | None = None) -> None:
        if self._inflight.is_set():
            return
        self._inflight.set()
        gen = self._gen   # tag this inference; discard its result if force_replan bumps _gen
        t = threading.Thread(
            target=self._background_fetch, args=(obs_fn, obs, gen), daemon=True, name="infer_bg"
        )
        t.start()

    def _background_fetch(self, obs_fn: Callable[[], dict], obs: dict | None = None, gen: int = 0) -> None:
        try:
            if obs is None:
                obs = obs_fn()   # capture fresh observation now, at inference start
            chunk = self._fetch(obs)
            with self._next_lock:
                if gen == self._gen:        # keep only if no force_replan happened meanwhile
                    self._next_chunk = chunk
                else:
                    logger.debug("Discarding stale background chunk (gen %d != %d).", gen, self._gen)
            logger.debug("Background inference complete.")
        except Exception:
            logger.exception("Background inference failed.")
        finally:
            self._inflight.clear()

    # ── Real-Time Chunking path ─────────────────────────────────────────────────────────
    def _get_action_rtc(self, obs_fn: Callable[[], dict]) -> np.ndarray:
        """RTC: continuous async execution. The next chunk is generated while the current one
        plays, conditioned (server-side inpainting) on the committed leftover so the seam is
        continuous. Chunks are aligned on an absolute timeline (_t); a returned chunk is spliced
        at the tick its observation was captured (its frozen prefix == what just executed)."""
        H = self._action_horizon

        # Cold start: blocking VANILLA inference (no prefix yet), then kick the first prefetch.
        if self._chunk is None:
            logger.info("RTC cold-start inference (blocking, no prefix)...")
            self._t = 0
            obs = obs_fn()
            self._chunk = self._fetch_rtc(obs, prefix=None, d=0)
            self._chunk_base_t = 0
            self._launch_rtc_fetch(obs_fn)
            return self._emit_rtc()

        # Splice in a ready prefetched chunk, aligned by the tick its obs was captured at.
        with self._next_lock:
            if self._next_chunk is not None:
                idx = self._t - self._next_base_t
                if 0 <= idx < H:
                    self._chunk = self._next_chunk
                    self._chunk_base_t = self._next_base_t
                    self._next_chunk = None
                    logger.debug("RTC splice at idx=%d (d=%d).", idx, self._rtc_d)
                    self._launch_rtc_fetch(obs_fn)   # immediately prefetch the next chunk
                else:
                    # Catastrophically slow/fast inference -> chunk out of range; drop & refetch.
                    self._next_chunk = None
                    self._launch_rtc_fetch(obs_fn)

        if self._t - self._chunk_base_t >= H:
            # Ran out before the next chunk was ready -> hold last action; ensure a fetch is queued.
            self._launch_rtc_fetch(obs_fn)
            return self._chunk[-1].copy()
        return self._emit_rtc()

    def _emit_rtc(self) -> np.ndarray:
        idx = int(np.clip(self._t - self._chunk_base_t, 0, self._action_horizon - 1))
        action = self._chunk[idx].copy()
        self._t += 1
        return action

    def _launch_rtc_fetch(self, obs_fn: Callable[[], dict]) -> None:
        if self._inflight.is_set():
            return
        self._inflight.set()
        base_t = self._t            # obs captured ~now corresponds to this timeline tick
        idx = self._t - self._chunk_base_t
        if self._chunk is not None and 0 <= idx < self._action_horizon:
            prefix = self._chunk[idx:].copy()    # committed leftover (absolute), (H-idx, 14)
        else:
            prefix = None
        gen = self._gen
        t = threading.Thread(target=self._rtc_bg,
                             args=(obs_fn, base_t, prefix, self._rtc_d, gen),
                             daemon=True, name="rtc_bg")
        t.start()

    def _rtc_bg(self, obs_fn: Callable[[], dict], base_t: int, prefix, d: int, gen: int) -> None:
        try:
            obs = obs_fn()
            chunk = self._fetch_rtc(obs, prefix=prefix, d=d)
            with self._next_lock:
                if gen == self._gen:
                    self._next_chunk = chunk
                    self._next_base_t = base_t
                else:
                    logger.debug("Discarding stale RTC chunk (gen %d != %d).", gen, self._gen)
        except Exception:
            logger.exception("RTC background inference failed.")
        finally:
            self._inflight.clear()

    def _fetch_rtc(self, obs: dict, prefix, d: int) -> np.ndarray:
        """Like _fetch but (a) attaches the RTC committed prefix + delay to the obs, and
        (b) measures the obs->chunk WALL-CLOCK to adapt d = ceil(EMA(wall) * inference_freq)."""
        if prefix is not None and len(prefix) > 0:
            obs = dict(obs)
            obs["rtc_prefix"] = np.asarray(prefix, dtype=np.float32)
            obs["rtc_delay"] = int(d)
        t0 = time.monotonic()
        with self._infer_lock:
            result = self._policy.infer(obs)
        wall = time.monotonic() - t0
        self._fetch_count += 1
        if self._inference_freq:
            self._rtc_wall_ema = (wall if self._rtc_wall_ema is None
                                  else 0.7 * self._rtc_wall_ema + 0.3 * wall)
            new_d = int(np.ceil(self._rtc_wall_ema * self._inference_freq))
            self._rtc_d = int(np.clip(new_d, 1, self._action_horizon - 1))
            if self._rtc_d > self._action_horizon // 2:
                logger.warning("[rtc] d=%d > H/2=%d: inference too slow for this control rate; "
                               "seam continuity degrades (lower --inference-freq or speed up the model).",
                               self._rtc_d, self._action_horizon // 2)
        st = result.get("server_timing") if isinstance(result, dict) else None
        if st is not None and "infer_ms" in st:
            self._last_infer_ms = float(st["infer_ms"])
            logger.info("[rtc] wall=%.2fs d=%d (server %.2fs)",
                        wall, self._rtc_d, self._last_infer_ms / 1000.0)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]
        return actions

    def force_replan(self) -> None:
        """Abandon the active (and pre-fetched) chunk so the next get_action() regenerates from a
        FRESH observation. Used by the client's force guard to drop a chunk that is pressing too
        hard — the next get_action cold-starts a blocking inference from the current (contact) pose."""
        with self._next_lock:
            self._gen += 1           # invalidate any in-flight (stale-obs) background inference
            self._chunk = None
            self._next_chunk = None
            self._step = 0
            self._anchor_base = None
            self._anchor_state = None

    def reset(self) -> None:
        with self._next_lock:
            self._chunk = None
            self._prev_chunk = None
            self._next_chunk = None
            self._step = 0
            self._anchor_base = None
            self._anchor_state = None
