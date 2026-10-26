# Sim

GPU-parallel MuJoCo Warp sim of the RC car, and PPO to train a driving policy in it.
See `../docs/sim-rl-plan.md` for the overall plan. Run everything from `sim/`.

**The policy drives from the camera only.** The actor is PilotNet (same layers as
`src/training/pilotnet.py`) on the car's 160x120 camera, preprocessed exactly like on
the Pi (`env/preprocess.py`: crop the top 30%, RGB). The critic sees sim-only state
(offset, heading, track ahead), but only during training; it never goes on the car.

## Train, watch, compare

```bash
uv run python train.py                            # experiments/baseline.toml
uv run python train.py experiments/smoother.toml  # an experiment (your own file)
uv run python view.py                             # watch the latest run drive
uv run tensorboard --logdir runs                  # compare runs in the browser
```

Each training run creates `runs/<name>_<date>_<time>/` with:

- `config.toml`: every setting the run used
- `actor.pt`: the trained policy
- TensorBoard curves

`view.py runs/<run>` watches a specific run, and `--stochastic` adds the exploration
noise used during training.

## Experiments

`experiments/baseline.toml` lists every knob with its value and a comment. It's the
only place defaults live. An experiment file is applied on top of it and only lists
what it changes:

```toml
# experiments/smoother.toml (example)
[reward]
steer_rate = 1.0
```

To try a value without making a file:

```bash
uv run python train.py --set reward.steer_rate=1.0 ppo.lr=1e-4 name=smoother_slow_lr
```

Misspelled keys and wrong types are errors, so a typo can't silently do nothing.

**A typical loop:** change one thing, train (a few minutes), watch it in `view.py`, compare the
curves against the baseline in TensorBoard, keep or revert.

## The reward

Reward per step is a weighted sum of terms (weights in `[reward]`):

| Term | Meaning |
| --- | --- |
| `progress` | distance driven along the track; full speed earns ~1 per step |
| `off_track` | one-time penalty for crossing the tape (ends the episode) |
| `center` | penalty × offset², offset 1 = on the tape |
| `steer_rate` | penalty × (steering change per step)² |

**Adding a new term** (e.g. a target speed) is a code change, made once:

1. add a weight to `[reward]` in `baseline.toml` (0 = off)
2. add the field to `RewardConfig` in `experiment.py` and `EnvParams` in `env/kernels.py`
3. compute it in the `step` kernel, add it to `reward`, and write it to `reward_terms`
4. copy it into `self.params` in `env/car_env.py`, and log it in `ppo/rollout.py`

After that it's just a knob.

## What the logs mean

| Metric | Meaning |
| --- | --- |
| `drive/laps_per_min` | how fast it gets around the track |
| `drive/crashes_per_min` | how often it leaves the track |
| `drive/speed` | average speed (max is 1.5 m/s) |
| `drive/offset_abs` | average distance from the center line, 1 = on the tape |
| `drive/steer_change` | average steering change per step (steering is -1 to 1): lower = smoother |
| `reward/*` | average per step of the total and of each term: which term is driving the behavior |
| `train/std_*` | exploration noise; it shrinks as the policy gets confident |

These are rates, not per-episode numbers, so they're meaningful for every rollout.

## Other scripts

- `drive.py`: drive the car yourself with an Xbox controller
- `bench.py`: env steps per second, with and without camera rendering
