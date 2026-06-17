"""Mock smoke test for the broker RTC path — NO server / NO robot.

Drives AsyncActionChunkBroker(rtc=True) with a mock policy that mimics the real RTC server:
it honours the committed prefix (server inpaints the frozen/overlap region). Verifies the broker
(a) never crashes over many ticks, (b) measures the inference delay d, (c) actually splices
(fetch_count grows), and (d) emits a CONTINUOUS action stream — a mis-aligned splice would show
up as a large step-to-step jump at the seam.
"""
import time

import numpy as np

from eval_utils.action_chunk_broker import AsyncActionChunkBroker

H = 24


class MockPolicy:
    def __init__(self, infer_s: float):
        self.infer_s = infer_s
        self.calls = 0

    def infer(self, obs: dict) -> dict:
        time.sleep(self.infer_s)          # simulate inference latency
        self.calls += 1
        # Smooth global ramp that drifts slightly each call (simulates a slowly-changing plan).
        start = 0.001 * self.calls
        chunk = (start + 0.0005 * np.arange(H))[:, None] * np.ones((1, 14), dtype=np.float32)
        chunk = chunk.astype(np.float32)
        prefix = obs.get("rtc_prefix")
        if prefix is not None:            # mimic the server: frozen/overlap prefix == committed
            L = min(len(prefix), H)
            chunk[:L] = np.asarray(prefix, dtype=np.float32)[:L]
        return {"actions": chunk, "server_timing": {"infer_ms": self.infer_s * 1000.0}}


def run(infer_s: float, freq: float, ticks: int = 200) -> float:
    pol = MockPolicy(infer_s)
    br = AsyncActionChunkBroker(policy=pol, action_horizon=H, rtc=True, inference_freq=freq)
    obs_fn = lambda: {"observation/state": np.zeros(14, np.float32)}
    period = 1.0 / freq
    acts = []
    for _ in range(ticks):
        acts.append(br.get_action(obs_fn))
        time.sleep(period)
    acts = np.asarray(acts)
    jumps = np.abs(np.diff(acts[:, 0]))   # joint-0 step-to-step delta
    print(f"  infer={infer_s:.2f}s freq={freq:>4.1f}Hz  ticks={ticks} fetches={pol.calls} "
          f"measured_d={br._rtc_d}  max_jump={jumps.max():.5f} mean_jump={jumps.mean():.5f}")
    return float(jumps.max())


if __name__ == "__main__":
    print("Broker RTC smoke test (mock policy honours committed prefix):")
    j1 = run(infer_s=0.6, freq=7.0)    # d~5, feasible (d <= H/2)
    j2 = run(infer_s=0.3, freq=10.0)   # d~3, comfortable
    worst = max(j1, j2)
    print("BROKER RTC SMOKE:", "PASS" if worst < 0.05 else "CHECK (seam jumps large)",
          f"(worst seam jump {worst:.5f})")
