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
uv run python eval.py                             # score it on 256 tracks it never trained on
uv run tensorboard --logdir runs                  # compare runs in the browser
```

Each training run creates `runs/<name>_<date>_<time>/` with:

- `config.toml`: every setting the run used
- `actor_iterNNNN_best.pt`: the policy at its best reward so far (what `view.py` loads)
- `actor_iterNNNN.pt`: a checkpoint every `checkpoint_every` iterations
  (`view.py --checkpoint actor_iter0100.pt`)
- `actor.pt`: the policy at the end of training (`view.py --last`)
- TensorBoard curves

`eval.py` is the real score: every test track at once, one 30 s episode each from a
fixed start, with the mean action (no exploration noise, like on the car). It reports
the share finished without crashing, laps/min, speed, and offset, and lists crashes by
track. `--train` scores training tracks instead, to compare. `--randomize` gives each
track a random car and look (like training) instead of the `car.toml` car and the
default look.

`view.py` drives held-out test track 0 by default, a track the policy never trained on.
`--test N` picks another test track, `--track N` a training track. It starts from the
same spot as `eval.py`, so a crash it lists replays exactly (`--random-start` to vary it).

`view.py runs/<run>` watches a specific run, and `--stochastic` adds the exploration
noise used during training.

New bests (average reward per step) are marked with `*` in the console.

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
| `drive/laps_per_min` | how fast it gets around its tracks (each lap counted on its own track's length) |
| `drive/crashes_per_min` | how often it leaves the track |
| `drive/speed` | average speed (max is 1.5 m/s) |
| `drive/offset_abs` | average distance from the center line, 1 = on the tape |
| `drive/steer_change` | average steering change per step (steering is -1 to 1): lower = smoother |
| `reward/*` | average per step of the total and of each term: which term is driving the behavior |
| `train/std_*` | exploration noise; it shrinks as the policy gets confident |

These are rates, not per-episode numbers, so they're meaningful for every rollout.

## Random tracks

Training uses `[tracks] count` random tracks (1024), generated at startup from the
experiment's seed. Every reset puts the car on a random one. A separate set of
**test tracks** is never trained on; `view.py` drives those by default.

`assets/track_gen.py` makes them: a random polygon with every corner rounded into
an arc, so they have straights, S-bends, and hairpins, but no curve tighter than the
car can drive. Its knobs are `[tracks]` in `baseline.toml`. Track N for a seed is
always the same track.

How 1024 worlds show different tracks with one MuJoCo model: the model has one shared
set of tape boxes, and `env/kernels.py write_tape` moves each world's copy of them
onto that world's track before rendering (the renderer reads per-world geom poses).

```bash
uv run python show_tracks.py                                # tracks 0-15, top-down -> tracks.png
uv run python show_tracks.py --first 16                     # the next 16
uv run python show_tracks.py --test                          # the test tracks
uv run python show_tracks.py --set 'tracks.corners=[8,20]'  # try a setting
uv run python drive.py --track 3                            # drive track 3 yourself
uv run python drive.py --test 3                             # or test track 3
```

## The car: delay, lag, and randomization

The sim car responds like a real one instead of instantly:

- **Latency:** a command reaches the wheels `control.latency_s` after its camera frame
  (capture + inference + I2C on the Pi).
- **Servo and motor lag:** steering and speed approach their targets gradually, with
  time constants `steering.response_time_s` and `throttle.response_time_s`.

These values in `config/car.toml` are guesses until measured on the car, so `[randomize]`
in `baseline.toml` gives every episode a slightly different car: latency, servo and motor
response, top speed, and max steering angle each scaled by a random factor, plus a small
servo trim (the car drifts a little when "straight"). The policy has to work across the
whole range, so it should handle the real car even if the guesses are off. The critic
sees the episode's car; the actor only sees the camera.

`eval.py` and `view.py` use the `car.toml` car exactly unless `eval.py --randomize`.

## Visual randomization

The camera image is randomized every episode too (`[visuals]` in `baseline.toml`), so
the policy learns "follow the tape" on any floor, not the sim's gray checker:

- **Floor:** one of 7 textures (`assets/textures.py`: checker, concrete, carpet, planks,
  tiles, speckle, flat), tinted a random color and scaled. Planks and tiles have lines
  that aren't tape.
- **Tape:** white half the time, otherwise any color, 2-5 cm wide. It's always kept
  brighter or darker than the floor so it stays visible.
- **Camera mount:** +-1 cm height, +-3 degrees tilt.
- **Image:** exposure, contrast, white balance, and per-frame sensor noise.

The look lives in the model's per-world arrays (floor texture and color, tape color and
width, camera pose), set by `env/kernels.py randomize_look` at each reset; the image
settings are applied in `postprocess`. With the default look, `postprocess` is exactly
the Pi's crop (`env/preprocess.py`).

```bash
uv run python show_views.py             # 24 random looks as the policy sees them -> views.png
uv run python show_views.py --default   # the default look
uv run python show_views.py --set visuals.noise=0.1   # try a setting
```

## Other scripts

- `drive.py`: drive the car yourself with an Xbox controller (`--track N` / `--test N` for a random track, default the oval)
- `bench.py`: env steps per second, with and without camera rendering
