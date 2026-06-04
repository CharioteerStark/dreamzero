#!/usr/bin/env python3
"""Generate a ~16s world-model rollout video from the CURRENT live ZED/ZMQ camera frame.

Purpose: sanity-check whether the world model still produces coherent video under the
*current real lighting* (vs the training set). Captures ONE live frame set from the 3 ZMQ
camera streams (+ live arm state), primes the causal world model, autoregressively rolls
out ~21 latent frames, and decodes to a ~16s mp4 — same kind of rollout we ran on the
training set, but conditioned on the live camera.

Run on the machine that reaches the Jetson cameras + arms and has a GPU (use a free GPU if
the serve is on another):

  CUDA_VISIBLE_DEVICES=0 python gen_video_live.py \
     --model_path ./checkpoints/adam_stage_a_merged_19000 \
     --zmq-host 192.222.10.10 \
     --left-arm-ip 192.168.10.22 --right-arm-ip 192.168.10.201 \
     --prompt "Pick up the yellow cube and place it on the pink circular pad." \
     --device cuda:0 --blocks 9 --output_dir results_live_rollout

Notes:
  - read-only on the arms (get_servo_angle); does NOT move anything.
  - --no-state uses a zero state if you don't want to connect the arms.
  - blocks=9 -> ~21 latent frames -> ~81 decoded frames -> ~16s @ 5 fps.
"""
import torch._dynamo
torch._dynamo.config.disable = True

import argparse, os, sys, time
import cv2, numpy as np, torch
import torch.distributed as dist
from einops import rearrange
from tianshou.data import Batch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deploy_adam import _ZmqCamSubscriber, _read_arm_state  # noqa: E402
from groot.vla.data.schema import EmbodimentTag  # noqa: E402
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy  # noqa: E402

EXPECTED_RES = (640, 360)  # (W, H), matches serve_wam handshake
FONT = cv2.FONT_HERSHEY_SIMPLEX


def build_obs_live(head, lw, rw, state14, prompt):
    """Build the GrootSimPolicy obs dict from live frames (head->top) + 14-D state."""
    def f(img):
        return cv2.resize(img, EXPECTED_RES, interpolation=cv2.INTER_AREA)[np.newaxis].astype(np.uint8)  # (1,H,W,3)
    s = np.asarray(state14, dtype=np.float64)
    return {
        "video.top": f(head), "video.left_wrist": f(lw), "video.right_wrist": f(rw),
        "state.left_joint_pos":    s[0:6].reshape(1, 6),
        "state.left_gripper_pos":  s[6:7].reshape(1, 1),
        "state.right_joint_pos":   s[7:13].reshape(1, 6),
        "state.right_gripper_pos": s[13:14].reshape(1, 1),
        "annotation.task": prompt,
    }


def save_mp4(path, frames_rgb, fps=5):
    frames = list(frames_rgb); h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in frames:
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    vw.release()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="./checkpoints/adam_stage_a_merged_19000")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--zmq-host", default="192.222.10.10")
    p.add_argument("--head-port", type=int, default=5566)
    p.add_argument("--left-wrist-port", type=int, default=5568)
    p.add_argument("--right-wrist-port", type=int, default=5569)
    p.add_argument("--left-arm-ip", default="192.168.10.22")
    p.add_argument("--right-arm-ip", default="192.168.10.201")
    p.add_argument("--no-state", action="store_true", help="use zero state instead of reading the arms")
    p.add_argument("--input-image", default=None,
                   help="Use a saved 2x2-grid conditioning image (e.g. results_live_rollout/live_conditioning.png) "
                        "instead of capturing from the ZED/ZMQ cameras. Splits it back into top/left_wrist/"
                        "right_wrist; runs fully offline (zero state, no robot needed).")
    p.add_argument("--prompt", default="Pick up the yellow cube and place it on the pink circular pad.")
    p.add_argument("--blocks", type=int, default=9, help="continuation blocks (9 -> ~21 latent ~= 16s @ 5fps)")
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--output_dir", default="results_live_rollout")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.input_image:
        # ── use a saved 2x2-grid image instead of the cameras (offline, no robot) ──
        bgr = cv2.imread(args.input_image)
        if bgr is None:
            raise FileNotFoundError(f"--input-image not readable: {args.input_image}")
        grid_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gh, gw = grid_rgb.shape[0] // 2, grid_rgb.shape[1] // 2
        head = grid_rgb[:gh, :gw]      # top-left  = top
        lw   = grid_rgb[gh:, :gw]      # bottom-left = left_wrist
        rw   = grid_rgb[:gh, gw:]      # top-right = right_wrist
        state14 = np.zeros(14)         # no robot -> zero state
        print(f"Input image: {args.input_image} {grid_rgb.shape} -> top/lw/rw {head.shape}; zero state")
    else:
        # ── capture one live frame set from the ZMQ cameras (head -> top) ──
        cams = {
            "top":         _ZmqCamSubscriber("top",         args.zmq_host, args.head_port),
            "left_wrist":  _ZmqCamSubscriber("left_wrist",  args.zmq_host, args.left_wrist_port),
            "right_wrist": _ZmqCamSubscriber("right_wrist", args.zmq_host, args.right_wrist_port),
        }
        for c in cams.values():
            c.start()
        print("Waiting for live frames from ZMQ cameras...")
        head = lw = rw = None
        t0 = time.time()
        while time.time() - t0 < 10.0:
            head = cams["top"].get_latest(1.0)
            lw = cams["left_wrist"].get_latest(1.0)
            rw = cams["right_wrist"].get_latest(1.0)
            if head is not None and lw is not None and rw is not None:
                break
            time.sleep(0.2)
        if head is None or lw is None or rw is None:
            raise RuntimeError("No live frames from ZMQ cameras (check --zmq-host/ports and the Jetson publisher).")
        print(f"Live frames: top={head.shape} left_wrist={lw.shape} right_wrist={rw.shape}")
        if args.no_state:
            state14 = np.zeros(14)
        else:
            from xarm.wrapper import XArmAPI
            la, ra = XArmAPI(args.left_arm_ip), XArmAPI(args.right_arm_ip)
            lj, lg = _read_arm_state(la, "left")
            rj, rg = _read_arm_state(ra, "right")
            state14 = np.array(lj + [lg] + rj + [rg], dtype=np.float64)
        print("State (14):", np.round(state14, 3))

    # save the conditioning grid used (so you can eyeball the input scene)
    H, W = 176, 320
    grid = np.zeros((2 * H, 2 * W, 3), np.uint8)
    grid[:H, :W] = cv2.resize(head, (W, H)); grid[H:, :W] = cv2.resize(lw, (W, H)); grid[:H, W:] = cv2.resize(rw, (W, H))
    cv2.imwrite(os.path.join(args.output_dir, "live_conditioning.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    # ── load model + autoregressive rollout from the live frame ──
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost"); os.environ.setdefault("MASTER_PORT", "29530")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
    policy = GrootSimPolicy(embodiment_tag=EmbodimentTag.ADAM, model_path=args.model_path, device=args.device)
    ah = policy.trained_model.action_head
    if len(ah.dit_step_mask) != int(getattr(ah, "num_inference_steps", 16)):
        ah.dit_step_mask = [True] * int(ah.num_inference_steps)

    def fresh():
        return Batch(obs=build_obs_live(head, lw, rw, state14, args.prompt))

    ah.rollout_no_reset = False
    with torch.inference_mode():
        _, acc = policy.lazy_joint_forward_causal(fresh())     # (B,C,3,H,W)
    ah.rollout_no_reset = True
    for k in range(args.blocks):
        with torch.inference_mode():
            _, blk = policy.lazy_joint_forward_causal(fresh(), latent_video=acc.detach())
        acc = torch.cat([acc, blk], dim=2)
        print(f"  block {k+1}/{args.blocks} -> {acc.shape[2]} latent frames")
    ah.rollout_no_reset = False

    with torch.inference_mode():
        fr = ah.vae.decode(acc, tiled=ah.tiled, tile_size=(ah.tile_size_height, ah.tile_size_width),
                           tile_stride=(ah.tile_stride_height, ah.tile_stride_width))
    fr = rearrange(fr, "B C T H W -> B T H W C")[0]
    pred = ((fr.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    save_mp4(os.path.join(args.output_dir, "live_rollout.mp4"), pred, args.fps)

    sel = np.linspace(0, len(pred) - 1, 6).astype(int)
    cells = []
    for fi in sel:
        im = cv2.cvtColor(pred[fi], cv2.COLOR_RGB2BGR)
        cv2.rectangle(im, (0, 0), (im.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(im, f"t={fi/args.fps:.1f}s", (5, 16), FONT, 0.5, (0, 255, 0), 1)
        cells.append(im)
    cv2.imwrite(os.path.join(args.output_dir, "live_contact_sheet.png"),
                np.vstack([np.hstack(cells[i:i + 2]) for i in range(0, len(cells), 2)]))
    if not args.input_image:
        for c in cams.values():
            c.stop()
    print(f"Saved {len(pred)} frames ({len(pred)/args.fps:.1f}s) + live_conditioning.png + contact sheet to "
          f"{os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
