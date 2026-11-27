# AGENTS.md / CLAUDE.md — Autonomous RC Car

Self-driving XiaoRGEEK F1 RC car on a Raspberry Pi 5, with a GPU-parallel MuJoCo Warp RL simulator (`sim/`) and a C++20/ONNX Runtime inference stack (`include/`, `src/`).

## Repository Layout

- `sim/` — **Active development** (branch `feature/sim`): GPU-parallel MuJoCo Warp `CarEnv` + asymmetric PPO (`sim/rcsim/`). See [sim/README.md](sim/README.md) and [notes/sim-rl-plan.md](notes/sim-rl-plan.md).
  - `sim/train.py`, `sim/eval.py`, `sim/view.py` — train, benchmark on held-out tracks, watch rollouts
  - `sim/tools/` — `show_tracks.py`, `show_views.py`, `drive.py`, `bench.py`
  - `sim/rcsim/` — installed editable package (`env/`, `model/`, `ppo/`, `assets/`, `config.py`, `experiment.py`, `checkpoints.py`, `paths.py`)
  - `sim/experiments/baseline.toml` — single source of truth for training/sim defaults
- `config/car.toml` — Shared vehicle parameters (camera pose/FOV, servo/throttle limits, latency, control rate) used by sim and car.
- `include/`, `src/`, `tests/`, `CMakeLists.txt` — C++20 Pi runtime (camera thread, ONNX Runtime inference in `src/pilotnet_onnx.cpp`, MJPEG stream, I2C steering, GPIO motors, hardware smoke tests). Expects `$HOME/onnxruntime` on the Pi.
- `src/training/`, `scripts/export_onnx.py` — Legacy supervised behavioral cloning (PilotNet, YCbCr) + ONNX export.

## Common Commands

### Sim (run from `sim/`, managed with `uv`)

```bash
cd sim
uv run python train.py                              # experiments/baseline.toml
uv run python train.py --no-graphs --set ppo.iterations=2 name=smoke
uv run python eval.py                               # 256 held-out test tracks (nominal)
uv run python eval.py --randomize --tracks 1024     # randomized car + look
uv run python view.py                               # watch latest run on test track 0
uv run python tools/bench.py                        # env throughput benchmark
ruff check . && pyright                             # lint + typecheck
```

### C++ Runtime (on Pi / hosts with OpenCV + `~/onnxruntime`)

```bash
cmake -S . -B build && cmake --build build -j
```

## Key Invariants & Gotchas

- **Asymmetric actor-critic:** Actor only sees camera pixels (`84×160` uint8 RGB, cropped top 30% in `rcsim/env/preprocess.py`) plus optional previous commands (`obs[:, 11:13]`). **Never** feed sim-only state (`obs[:, 3:5]` true speed/steer, track offset, etc.) to the actor — the real car has no speed or steering sensors.
- **RGB vs YCbCr:** The RL actor trains on plain **RGB** (crop top 30% only, `/255`). The legacy BC model (`src/training/`, `scripts/export_onnx.py`) used YCbCr — do not mix them when exporting RL models to ONNX.
- **Hardware protocols (XiaoRGEEK F1 / PWR.A53.A / Pi 5):**
  - Steering (I2C `0x17`, cmd `0xFF`, `[channel=1, value]`): `0` = left, `85` = center, `170` = right. Send commands **3×** (controller one-command buffer quirk). Servo stays locked on exit until power-cycled.
  - Motors (GPIO): PWM `13`, `IN1=19`, `IN2=16` (Forward `HIGH/LOW`, Reverse `LOW/HIGH`). Always test with wheels off the ground.
  - Sign convention when deploying sim → car: verify steering sign flip (`sim +1 = left` vs car `+1 = right`) and `throttle = -1` = stop.
- **GPU hygiene:** Before running GPU-heavy commands in `sim/`, check `pgrep -af train.py` / `nvidia-smi` so you don't collide with an active training run.
- **Workflow preferences:** Keep changes in small, well-commented steps with big-picture explanations. For model architecture and recurrent-PPO code (`sim/rcsim/model/`, recurrent plumbing), guide and review rather than writing it unless asked. Commit/push only when asked.
