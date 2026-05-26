# Inference Commands

## 1. Start the Adam Inference Server (H100 machine)

```bash
cd ~/tony/dreamzero
bash scripts/inference/serve_wam.sh \
    ./checkpoints/adam_stage_a_lora/checkpoint-5000 5000 2 0,1
```

Wait for the log line: `Waiting for connections`

- First inference: ~30s (torch.compile warmup). After warmup: ~3s on H100.
- After retraining, replace `checkpoint-5000` with the new checkpoint path.

---

## 2. Run the Real Robot (deploy_adam.py)

Runs on any machine that can reach the H100 server and the Jetson.

```bash
cd ~/tony/dreamzero

# Dry-run (no motion — logs targets only):
python deploy_adam.py --prompt "pick up the cube"

# Live motion:
python deploy_adam.py --no-dry-run --prompt "pick up the cube"
```

**Key defaults:**
| Flag | Default | Description |
|------|---------|-------------|
| `--policy-host` | `localhost` | IP of the H100 running serve_wam.py |
| `--policy-port` | `5000` | Port of serve_wam.py |
| `--zmq-host` | `192.222.10.10` | Jetson IP (pub_zed.py ZMQ camera feeds) |
| `--left-arm-ip` | `192.168.10.22` | Left xArm 6 |
| `--right-arm-ip` | `192.168.10.201` | Right xArm 6 |
| `--inference-freq` | `10.0 Hz` | Control loop frequency |
| `--chunk-size` | `24` | Action horizon (must match model) |
| `--max-joint-jump-deg` | `30°` | Safety limit — skips action if exceeded |

---

## 3. Test Client (no robot hardware needed)

Validates the inference server end-to-end using pre-recorded debug frames.

```bash
# Requires socket_test_optimized_AR.py server (see below)
python test_client_AR.py --port 5000
```

### Start the roboarena test server (alternative to serve_wam.py):

```bash
CONDA_ENV=/home/thematrix/miniconda3/envs/dreamzero && \
CUDA_VISIBLE_DEVICES=0,1 \
CUDA_HOME=$CONDA_ENV \
PATH=$CONDA_ENV/bin:$PATH \
$CONDA_ENV/bin/torchrun --standalone --nproc_per_node=2 \
    socket_test_optimized_AR.py \
    --port 5000 \
    --embodiment adam \
    --model_path ./checkpoints/adam_stage_a_lora/checkpoint-5000 \
    --enable_dit_cache
```

---

## Notes

- **serve_wam.py** is the production server used by `deploy_adam.py` (msgpack protocol, server-side image resize to 640×360).
- **socket_test_optimized_AR.py** is the test server used by `test_client_AR.py` (roboarena protocol). Do not mix the two.
- Jetson must be running `pub_zed.py` before starting `deploy_adam.py`.
- `deploy_adam.py` captures the current arm position as the safe home on startup — position Adam safely before running.
