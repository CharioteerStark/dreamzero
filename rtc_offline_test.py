"""Offline RTC equivalence test — NO robot required.

Connects to a running serve_wam.py (RTC-enabled build) and checks that the Real-Time-Chunking
inpainting round-trips: when we hand the server a committed prefix and ask it to freeze the first
`d` steps, the FROZEN JOINT prefix of the returned chunk must reproduce that committed prefix.

Why this works: the model is deterministically seeded, so two infers on the SAME observation are
identical. C0 = unapply(model_norm) is the absolute chunk. Feeding rtc_prefix=C0 makes the server
reconstruct target_norm = q99(C0 - last_state) = model_norm (a round-trip), pin the prefix to it,
and unapply back to C0. So C1[:d, joints] ≈ C0[:d, joints]. It exercises the full server path on
both tensor-parallel ranks (a desync would deadlock instead of returning).

Usage (after restarting serve_wam.py with RTC support):
    python rtc_offline_test.py --port 5000 --freeze 5
"""
import argparse

import numpy as np

from eval_utils.wam_client import WamClientPolicy

# 12 joint dims of the 14-D action (grippers 6,13 are NOT inpainted by RTC).
JOINT_IDX = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def make_obs(prompt: str) -> dict:
    rng = np.random.default_rng(0)
    def img():
        return rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
    return {
        "observation/head_left": img(),
        "observation/left_wrist": img(),
        "observation/right_wrist": img(),
        # Zero state -> absolute == relative delta, simplest exact round-trip.
        "observation/state": np.zeros(14, dtype=np.float32),
        "prompt": prompt,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--freeze", type=int, default=5, help="d: leading steps to hard-freeze")
    ap.add_argument("--tol", type=float, default=2e-2, help="max |C1-C0| on frozen joints (rad)")
    args = ap.parse_args()

    cli = WamClientPolicy(host=args.host, port=args.port)
    obs = make_obs("place the orange cube on the white pad")

    print("[1/3] vanilla inference (no prefix) -> C0 ...")
    C0 = np.asarray(cli.infer(obs)["actions"], dtype=np.float32)
    print(f"      C0 shape {C0.shape}, joint range [{C0[:, JOINT_IDX].min():.3f}, "
          f"{C0[:, JOINT_IDX].max():.3f}]")

    print("[2/3] determinism check (same obs again) ...")
    C0b = np.asarray(cli.infer(obs)["actions"], dtype=np.float32)
    det = float(np.abs(C0b - C0).max())
    print(f"      max|C0b-C0| = {det:.6f}  ({'deterministic' if det < 1e-3 else 'NON-DETERMINISTIC!'})")

    print(f"[3/3] RTC inference with rtc_prefix=C0, rtc_delay={args.freeze} -> C1 ...")
    obs2 = dict(obs)
    obs2["rtc_prefix"] = C0.copy()
    obs2["rtc_delay"] = args.freeze
    C1 = np.asarray(cli.infer(obs2)["actions"], dtype=np.float32)
    print(f"      C1 shape {C1.shape}")

    d = args.freeze
    frozen = np.abs(C1[:d][:, JOINT_IDX] - C0[:d][:, JOINT_IDX])
    tail = np.abs(C1[d:][:, JOINT_IDX] - C0[d:][:, JOINT_IDX])
    print(f"      frozen joints [0:{d}]  max|C1-C0| = {frozen.max():.5f} rad (mean {frozen.mean():.5f})")
    print(f"      free   joints [{d}:]   max|C1-C0| = {tail.max():.5f} rad (mean {tail.mean():.5f})")

    ok = frozen.max() < args.tol
    print("\nRTC FROZEN-PREFIX ROUND-TRIP:", "PASS" if ok else "FAIL", f"(tol {args.tol} rad)")
    print("(free-region difference is expected and harmless; only the frozen prefix must match.)")
    cli.close()


if __name__ == "__main__":
    main()
