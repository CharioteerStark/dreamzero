"""WAM inference server for the Adam bimanual policy.

This file is the SERVER side only. It loads a WAN-based GrootSimPolicy and
serves it over an openpi-style websocket wire protocol (msgpack-encoded
observation/action dicts). No real-robot client is bundled with this repo;
any client that implements the protocol below can connect.

Architecture
------------
serve_wam.py (this file)
    └─ Wraps GrootSimPolicy (WAM 14B causal DiT) in a websocket policy server
    └─ Distributed across N GPUs via torchrun (tensor parallel; ~1.7x on 2 GPUs)

Wire protocol
-------------
Connection: ws://<host>:<port>
First server frame: msgpack-encoded metadata dict with keys
    model_name, model_path, embodiment, action_horizon=24, action_dim=14,
    state_dim=14, image_keys, state_layout, expected_image_resolution,
    world_size, default_prompt

Per-step request (client -> server), msgpack dict:
    observation/head_left    : (H, W, 3) uint8 RGB  (resized to 640x360 server-side)
    observation/left_wrist   : (H, W, 3) uint8 RGB  (resized to 640x360 server-side)
    observation/right_wrist  : (H, W, 3) uint8 RGB  (resized to 640x360 server-side)
    observation/state        : (14,) float32 =
                               [L-arm 6 joints, L-grip 1, R-arm 6 joints, R-grip 1]
    prompt                   : str

Per-step response (server -> client), msgpack dict:
    actions       : (24, 14) float32 = [L-arm(6), L-grip(1), R-arm(6), R-grip(1)] per step
    server_timing : {"infer_ms": float, "prev_total_ms": float}

Usage
-----
    # 2-GPU tensor parallel (recommended)
    bash scripts/inference/serve_wam.sh \\
        ./checkpoints/adam_stage_a_lora/checkpoint-5000 5000 2 0,1
"""

import asyncio
import dataclasses
import datetime
import http
import logging
import os
import pickle
import queue
import socket
import threading
import time
import traceback

import cv2
import numpy as np
import torch
import torch.distributed as dist
import tyro
import websockets.asyncio.server as _server
import websockets.frames
from openpi_client import msgpack_numpy
from tianshou.data import Batch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

logger = logging.getLogger(__name__)


# Signal codes broadcast from rank 0 to worker ranks via the gloo signal group.
_SIGNAL_CONTINUE = 0
_SIGNAL_SHUTDOWN = 1
_SIGNAL_IDLE = 2


@dataclasses.dataclass
class Args:
    """CLI arguments for the WAM inference server."""

    # Websocket port the policy client connects to.
    port: int = 8000
    # Path to the trained checkpoint directory (must contain experiment_cfg/, model.safetensors).
    model_path: str = "./checkpoints/adam_stage_a_lora/checkpoint-5000"
    # Enable KV caching during diffusion denoising (skip ~half of DiT forward passes).
    enable_dit_cache: bool = True
    # Used when the client sends an empty prompt; drawn from data/meta/tasks.jsonl.
    default_prompt: str = "Pick up the yellow cube and place it on the pink circular pad."
    # Timeout for the distributed signal group (gloo). 50000s ~= 14h, enough for any session.
    timeout_seconds: int = 50000
    # Expected WAM camera width/height (input is resized automatically if mismatched).
    image_width: int = 640
    image_height: int = 360
    # Directory to save world-model video predictions for each replan. Empty = DISABLED (default).
    # When set, each replan VAE-decodes the predicted latents to an MP4 (GPU work on a background
    # thread + disk I/O) — extra load during serving, so off by default. Pass --save-video-dir to enable.
    save_video_dir: str = ""


class AdamWanPolicy:
    """Wraps the WAN GrootSimPolicy and adapts it to the server's I/O schema.

    The model was trained with bimanual training keys (video.top/left_wrist/right_wrist,
    state.left_joint_pos/..., action.left_joint_pos/...). This class translates the
    client-facing observation keys (see module docstring) into those training keys,
    and re-flattens the model's 4 action heads into the 14-D action vector returned
    on the wire.

    Distributed: on multi-GPU runs, this class is instantiated only on rank 0.
    Worker ranks run worker_loop() and receive the obs via broadcast.
    """

    # Adam's eval_delta_indices == [0]: a single current frame per call.
    # The WAN DiT KV cache handles temporal context internally.
    FRAMES_PER_CHUNK = 1

    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        signal_group: "dist.ProcessGroup",
        default_prompt: str = "",
        expected_image_resolution: tuple[int, int] = (640, 360),
        save_video_dir: str = "",
    ) -> None:
        self._policy = groot_policy
        self._signal_group = signal_group
        self._default_prompt = default_prompt
        self._required_image_keys = (
            "observation/head_left",
            "observation/left_wrist",
            "observation/right_wrist",
        )
        self._expected_image_resolution = expected_image_resolution
        self._save_video_dir = save_video_dir
        self._replan_count = 0
        self._save_queue: "queue.Queue | None" = None
        if save_video_dir:
            os.makedirs(save_video_dir, exist_ok=True)
            self._save_queue = queue.Queue()
            t = threading.Thread(target=self._video_save_worker, daemon=True, name="video_saver")
            t.start()
            logger.info("World-model video saving enabled → %s", save_video_dir)

    def _resize_if_needed(self, frame: np.ndarray, key: str) -> np.ndarray:
        expected_w, expected_h = self._expected_image_resolution
        h, w = frame.shape[:2]
        if (w, h) == (expected_w, expected_h):
            return frame
        logger.warning(
            "Resizing %s from (%d, %d) to (%d, %d) for WAM inference",
            key,
            w,
            h,
            expected_w,
            expected_h,
        )
        return cv2.resize(frame, (expected_w, expected_h), interpolation=cv2.INTER_AREA)

    def _convert_observation(self, obs: dict) -> dict:
        """Wire-protocol obs keys -> WAN training keys."""
        converted: dict = {}

        # 3 cameras: wire names -> training names (Adam shares YAM's projector index 32,
        # so [top, left_wrist, right_wrist] must match that stitching order).
        image_key_mapping = {
            "observation/head_left":  "video.top",
            "observation/left_wrist": "video.left_wrist",
            "observation/right_wrist": "video.right_wrist",
        }
        for wire_key in self._required_image_keys:
            if obs.get(wire_key) is None:
                raise ValueError(f"Missing required WAM image key: {wire_key}")

        for wire_key, wan_key in image_key_mapping.items():
            frame = obs[wire_key]
            if not isinstance(frame, np.ndarray):
                frame = np.asarray(frame)
            if frame.ndim == 3:
                # Single (H, W, 3) -> add a leading time dim.
                if frame.shape[-1] != 3:
                    raise ValueError(f"Expected {wire_key} channels=3, got shape {frame.shape}")
                frame = self._resize_if_needed(frame, wire_key)
                converted[wan_key] = frame[np.newaxis].astype(np.uint8)
            elif frame.ndim == 4:
                # Already (T, H, W, 3) -> use the most recent frame.
                frame = frame[-1:]
                if frame.shape[-1] != 3:
                    raise ValueError(f"Expected {wire_key} channels=3, got shape {frame.shape}")
                converted[wan_key] = np.stack(
                    [self._resize_if_needed(f, wire_key) for f in frame], axis=0
                ).astype(np.uint8)
            else:
                raise ValueError(
                    f"Unexpected shape for {wire_key}: {frame.shape}"
                )

        # State layout (per wire protocol; see module docstring):
        #   state[0:6]   = left arm joints (radians)
        #   state[6]     = left gripper (raw/10)
        #   state[7:13]  = right arm joints
        #   state[13]    = right gripper
        state = obs.get("observation/state")
        if state is None:
            raise ValueError("Missing required WAM state key: observation/state")
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        if state.size != 14:
            raise ValueError(
                f"WAM server expects observation/state with 14 values, got {state.size}"
            )

        converted["state.left_joint_pos"]    = state[0:6].reshape(1, 6)
        converted["state.left_gripper_pos"]  = state[6:7].reshape(1, 1)
        converted["state.right_joint_pos"]   = state[7:13].reshape(1, 6)
        converted["state.right_gripper_pos"] = state[13:14].reshape(1, 1)

        # Language: Adam training used annotation.task, not annotation.language.action_text.
        prompt = obs.get("prompt") or self._default_prompt
        converted["annotation.task"] = prompt

        return converted

    def _convert_action(self, action_dict: dict) -> np.ndarray:
        """4 named action tensors -> flat (N, 14) array matching state layout."""
        keys_in_order = [
            ("action.left_joint_pos",    6),
            ("action.left_gripper_pos",  1),
            ("action.right_joint_pos",   6),
            ("action.right_gripper_pos", 1),
        ]
        parts: list[np.ndarray] = []
        N: int | None = None

        for key, _dim in keys_in_order:
            v = action_dict.get(key)
            if v is None:
                parts.append(None)  # placeholder, filled once N is known
                continue
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            v = np.asarray(v)
            if v.ndim == 0:
                v = v.reshape(1, 1)
            elif v.ndim == 1:
                v = v.reshape(-1, 1)
            parts.append(v)
            if N is None:
                N = v.shape[0]

        if N is None:
            # Model returned nothing recognizable; emit a safe zero chunk.
            return np.zeros((1, 14), dtype=np.float32)

        filled = []
        for (_key, dim), part in zip(keys_in_order, parts):
            if part is None:
                filled.append(np.zeros((N, dim), dtype=np.float32))
            else:
                filled.append(part)
        return np.concatenate(filled, axis=-1).astype(np.float32)

    def _broadcast_obs_to_workers(self, obs: dict) -> None:
        """Pickle obs and broadcast to non-rank-0 ranks so they can run the same forward pass."""
        serialized = pickle.dumps(obs)
        size_tensor = torch.tensor([len(serialized)], dtype=torch.int64, device="cuda")
        dist.broadcast(size_tensor, src=0)
        data_tensor = torch.frombuffer(serialized, dtype=torch.uint8).cuda()
        dist.broadcast(data_tensor, src=0)

    def infer(self, obs: dict) -> dict:
        """Inference entry point. Returns a dict; the transport adds server_timing."""
        converted_obs = self._convert_observation(obs)

        # Signal worker ranks to participate in this forward pass.
        signal_tensor = torch.tensor([_SIGNAL_CONTINUE], dtype=torch.int32, device="cpu")
        dist.broadcast(signal_tensor, src=0, group=self._signal_group)
        self._broadcast_obs_to_workers(converted_obs)

        batch = Batch(obs=converted_obs)
        dist.barrier()
        with torch.no_grad():
            result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
        dist.barrier()

        # Enqueue world-model video pred + the ACTUAL observed grid for background save.
        if self._save_queue is not None and video_pred is not None:
            self._replan_count += 1
            self._save_queue.put((self._replan_count, video_pred.detach().cpu(),
                                  self._actual_grid(converted_obs)))

        # result_batch.act is a tianshou Batch with action.* attributes.
        action_chunk = result_batch.act
        action_dict = {
            k: getattr(action_chunk, k) for k in dir(action_chunk) if k.startswith("action.")
        }
        actions = self._convert_action(action_dict)
        # Shape: (action_horizon=24, 14).
        return {"actions": actions}

    def _actual_grid(self, converted_obs: dict) -> np.ndarray:
        """Build the 2x2 grid of the ACTUAL observed frame (RGB), matching the model's layout
        (top->TL, left_wrist->BL, right_wrist->TR, BR black)."""
        H, W = 176, 320
        top = converted_obs["video.top"][0]
        lw = converted_obs["video.left_wrist"][0]
        rw = converted_obs["video.right_wrist"][0]
        g = np.zeros((2 * H, 2 * W, 3), np.uint8)
        g[:H, :W] = cv2.resize(top, (W, H)); g[H:, :W] = cv2.resize(lw, (W, H)); g[:H, W:] = cv2.resize(rw, (W, H))
        return g

    def _video_save_worker(self) -> None:
        """Background thread: drain the save queue, decode VAE latents, write MP4 + actual frame."""
        while True:
            item = self._save_queue.get()
            if item is None:
                break
            replan_idx, video_pred_cpu, actual_grid = item
            try:
                self._decode_and_save(replan_idx, video_pred_cpu, actual_grid)
            except Exception:
                logger.exception("World-model video save failed for replan %d", replan_idx)
            finally:
                self._save_queue.task_done()

    def _decode_and_save(self, replan_idx: int, video_pred_cpu: torch.Tensor,
                         actual_grid: "np.ndarray | None" = None) -> None:
        """Decode VAE latents to pixels and write an MP4. video_pred_cpu is on CPU.
        Also save the actual observed grid (PNG) so predicted vs actual can be compared."""
        ah = self._policy.trained_model.action_head
        device = next(ah.vae.parameters()).device

        with torch.no_grad():
            # video_pred is (B, T, C, H, W) latent — vae.decode expects same.
            frames_bcthw = ah.vae.decode(
                video_pred_cpu.to(device=device),
                tiled=ah.tiled,
                tile_size=(ah.tile_size_height, ah.tile_size_width),
                tile_stride=(ah.tile_stride_height, ah.tile_stride_width),
            )
        # (B, C, T, H, W) -> (T, H, W, C) uint8 RGB
        frames = frames_bcthw[0].permute(1, 2, 3, 0)
        frames = ((frames.float() + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self._save_video_dir, f"replan_{replan_idx:04d}_{ts}.mp4")
        h, w = frames.shape[1], frames.shape[2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        # Save the ACTUAL observed grid for this replan (paired by index for side-by-side).
        if actual_grid is not None:
            cv2.imwrite(os.path.join(self._save_video_dir, f"actual_{replan_idx:04d}_{ts}.png"),
                        cv2.cvtColor(actual_grid, cv2.COLOR_RGB2BGR))
        logger.info("Saved world-model video: %s (%d frames) + actual frame", path, len(frames))

    def reset(self) -> None:
        """Protocol has no reset wire message; this is a no-op.

        Adam uses eval_delta_indices=[0] so each call is self-contained
        (KV cache is reset every call). Nothing to do here.
        """
        return


class WamWebsocketServer:
    """Minimal websocket transport for the WAM policy.

    Implements the openpi-style wire protocol (msgpack-encoded request/response,
    one metadata frame on connect). Vendored in this file so the repo does not
    pull in the full openpi server package.
    """

    def __init__(
        self,
        policy: AdamWanPolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=self._health_check,
            ping_interval=None,
        ) as server:
            await server.serve_forever()

    @staticmethod
    def _health_check(connection: _server.ServerConnection, request: _server.Request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        # 1. metadata handshake
        await websocket.send(packer.pack(self._metadata))

        prev_total_time: float | None = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_start = time.monotonic()
                action = self._policy.infer(obs)
                infer_ms = (time.monotonic() - infer_start) * 1000.0

                action["server_timing"] = {"infer_ms": infer_ms}
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def init_mesh() -> DeviceMesh:
    """Initialize the distributed process group and a 1-D device mesh for tensor parallel."""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    logger.info(f"Rank {rank}/{world_size} (PID {os.getpid()}) bound to cuda:{rank}")
    mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size,),
        mesh_dim_names=("ip",),
    )
    return mesh


def worker_loop(policy: GrootSimPolicy, signal_group: "dist.ProcessGroup") -> None:
    """Non-rank-0 worker loop: wait for broadcast, run forward pass, repeat."""
    rank = dist.get_rank()
    signal_tensor = torch.zeros(1, dtype=torch.int32, device="cpu")
    logger.info(f"Worker loop started on rank {rank}")

    while True:
        try:
            dist.broadcast(signal_tensor, src=0, group=signal_group)
            sig = int(signal_tensor.item())
            if sig == _SIGNAL_SHUTDOWN:
                logger.info(f"Rank {rank} received shutdown signal")
                break
            if sig == _SIGNAL_IDLE:
                continue

            # Receive obs.
            size_tensor = torch.zeros(1, dtype=torch.int64, device="cuda")
            dist.broadcast(size_tensor, src=0)
            data_size = int(size_tensor.item())
            data_tensor = torch.zeros(data_size, dtype=torch.uint8, device="cuda")
            dist.broadcast(data_tensor, src=0)
            obs = pickle.loads(data_tensor.cpu().numpy().tobytes())

            batch = Batch(obs=obs)
            dist.barrier()
            with torch.no_grad():
                _ = policy.lazy_joint_forward_causal(batch)
            dist.barrier()

        except Exception as e:
            logger.error(f"Worker loop error on rank {rank}: {e}")
            traceback.print_exc()
            break


def main(args: Args) -> None:
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    os.environ["ATTENTION_BACKEND"] = "TE"
    # Inference is autoregressive across chunks — give Dynamo room to compile multiple shapes.
    torch._dynamo.config.recompile_limit = 800

    device_mesh = init_mesh()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Separate signal group on gloo so rank 0's CPU-side ops don't share the NCCL stream.
    timeout_delta = datetime.timedelta(seconds=args.timeout_seconds)
    signal_group = dist.new_group(backend="gloo", timeout=timeout_delta)

    # Load the WAN policy on every rank (parallelize() shards tensors across the device mesh).
    groot_policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag("adam"),
        model_path=args.model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
    )

    if rank == 0:
        wrapper = AdamWanPolicy(
            groot_policy=groot_policy,
            signal_group=signal_group,
            default_prompt=args.default_prompt,
            expected_image_resolution=(args.image_width, args.image_height),
            save_video_dir=args.save_video_dir,
        )

        hostname = socket.gethostname()
        try:
            local_ip = socket.gethostbyname(hostname)
        except socket.gaierror:
            local_ip = "0.0.0.0"

        metadata = {
            "model_name": "dreamzero_wan_adam",
            "model_path": args.model_path,
            "embodiment": "adam",
            "action_horizon": 24,
            "action_dim": 14,
            "state_dim": 14,
            "image_keys": [
                "observation/head_left",
                "observation/left_wrist",
                "observation/right_wrist",
            ],
            "state_layout": "[L-arm(6), L-grip(1), R-arm(6), R-grip(1)]",
            "expected_image_resolution": [args.image_width, args.image_height],
            "world_size": world_size,
            "default_prompt": args.default_prompt,
        }

        logger.info(
            "WAM inference server ready: host=%s ip=%s port=%d world_size=%d",
            hostname, local_ip, args.port, world_size,
        )
        logger.info("Clients should set chunk_size=24 (model action_horizon).")

        server = WamWebsocketServer(
            policy=wrapper,
            host="0.0.0.0",
            port=args.port,
            metadata=metadata,
        )
        server.serve_forever()
    else:
        worker_loop(groot_policy, signal_group)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        force=True,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main(tyro.cli(Args))
