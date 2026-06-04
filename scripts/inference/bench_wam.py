#!/usr/bin/env python3
"""Latency benchmark for serve_wam.py.

Connects via the production wam_client, sends N synthetic observations
(zeros for images + state), measures per-call latency, prints stats.

The server's DiT KV cache is reset between calls (model design), so each
call is independent — a fair single-shot inference measurement.

Usage:
    python scripts/inference/bench_wam.py --host localhost --port 5000 --n 30 --warmup 3
"""

import argparse
import logging
import statistics
import time

import numpy as np

from eval_utils.wam_client import WamClientPolicy


def synthetic_obs(prompt: str) -> dict:
    """Build a zero-valued observation matching serve_wam's expected schema."""
    return {
        "observation/head_left":   np.zeros((360, 640, 3), dtype=np.uint8),
        "observation/left_wrist":  np.zeros((360, 640, 3), dtype=np.uint8),
        "observation/right_wrist": np.zeros((360, 640, 3), dtype=np.uint8),
        "observation/state":       np.zeros((14,), dtype=np.float32),
        "prompt":                  prompt,
    }


def percentile(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--n", type=int, default=30, help="Number of timed calls.")
    p.add_argument("--warmup", type=int, default=3, help="Untimed warmup calls.")
    p.add_argument("--prompt", default="Pick up the yellow cube and place it on the pink circular pad.")
    p.add_argument("--label", default="run", help="Label to print in the summary line.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("bench")

    policy = WamClientPolicy(host=args.host, port=args.port)
    log.info("Metadata: %s", policy.get_server_metadata())

    obs = synthetic_obs(args.prompt)

    log.info("Warmup x%d (untimed, includes torch.compile warmup on first call)...", args.warmup)
    for i in range(args.warmup):
        t0 = time.perf_counter()
        _ = policy.infer(obs)
        log.info("  warmup %d: %.3fs", i, time.perf_counter() - t0)

    log.info("Timed x%d ...", args.n)
    latencies = []
    server_times = []
    for i in range(args.n):
        t0 = time.perf_counter()
        resp = policy.infer(obs)
        dt = time.perf_counter() - t0
        latencies.append(dt)
        st = resp.get("server_timing")
        if isinstance(st, dict):
            for k, v in st.items():
                if isinstance(v, (int, float)):
                    server_times.append((k, v))
        log.info("  call %02d: %.3fs", i, dt)

    policy.close()

    print()
    print("================ BENCH SUMMARY ================")
    print(f"label        : {args.label}")
    print(f"n            : {args.n}")
    print(f"mean         : {statistics.mean(latencies):.3f} s")
    print(f"median       : {statistics.median(latencies):.3f} s")
    print(f"std          : {statistics.pstdev(latencies):.3f} s")
    print(f"p50          : {percentile(latencies, 50):.3f} s")
    print(f"p95          : {percentile(latencies, 95):.3f} s")
    print(f"p99          : {percentile(latencies, 99):.3f} s")
    print(f"min / max    : {min(latencies):.3f} / {max(latencies):.3f} s")
    print(f"throughput   : {args.n / sum(latencies):.2f} calls/s")
    if server_times:
        keys = sorted({k for k, _ in server_times})
        for k in keys:
            vs = [v for kk, v in server_times if kk == k]
            print(f"server.{k:<20} mean={statistics.mean(vs):.4f}  p95={percentile(vs,95):.4f}  n={len(vs)}")
    print("===============================================")


if __name__ == "__main__":
    main()
