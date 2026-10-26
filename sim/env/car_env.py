"""GPU-parallel car env, the car version of parallel-cartpole's CartPole.

Each world drives its own track, picked at random from `tracks` on every reset.
Actions are [steer, throttle] in [-1, 1].

Two kinds of observation:
- pixels (render=True): the car's camera, preprocessed exactly like on the Pi
  (env/preprocess.py). This is all the actor sees.
- obs: sim-only state (env/kernels.py:write_obs). Only the critic sees it, during
  training; the real car never has it.
"""

from dataclasses import dataclass

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from assets.car import build_model
from assets.track import Track, oval_centerline, tape_poses
from config import load_config
from env.kernels import (
    NUM_REWARD_TERMS,
    OBS_DIM,
    CarParams,
    EnvParams,
    TrackBank,
    reset_all,
    step,
    write_tape,
)
from env.preprocess import crop_top, preprocess
from experiment import RewardConfig, load_experiment


@dataclass
class StepOutput:
    obs: torch.Tensor
    rgb: torch.Tensor | None  # raw render, (N, 3, H, W) float in [0, 1]
    pixels: torch.Tensor | None  # what the actor sees, (N, 3, H', W) uint8 RGB
    reward: torch.Tensor
    done: torch.Tensor
    truncated: torch.Tensor
    final_obs: torch.Tensor


@dataclass
class ResetOutput:
    obs: torch.Tensor
    rgb: torch.Tensor | None
    pixels: torch.Tensor | None


def _pack(rows: list[np.ndarray], dtype) -> wp.array:
    """Rows of different lengths -> one (len(rows), longest) array, zero padded."""
    out = np.zeros(
        (len(rows), max(len(r) for r in rows), *rows[0].shape[1:]), np.float32
    )
    for k, r in enumerate(rows):
        out[k, : len(r)] = r
    return wp.array(out, dtype=dtype)


def make_track_bank(tracks: list[Track]) -> tuple[TrackBank, int]:  # pyright: ignore[reportGeneralTypeIssues]
    """Tracks -> GPU arrays for the kernels, and how many tape boxes the longest
    track needs."""
    tapes = [tape_poses(t) for t in tracks]
    bank = TrackBank()
    bank.centerline = _pack([t.centerline for t in tracks], wp.vec2)
    bank.tangents = _pack([t.tangents for t in tracks], wp.vec2)
    bank.normals = _pack([t.normals for t in tracks], wp.vec2)
    bank.s = _pack([t.s for t in tracks], float)
    bank.n = wp.array([len(t.centerline) for t in tracks], dtype=int)
    bank.length = wp.array([t.length for t in tracks], dtype=float)
    bank.half_width = wp.array([t.width / 2 for t in tracks], dtype=float)
    bank.tape = _pack(tapes, wp.vec3)
    bank.n_tape = wp.array([len(p) for p in tapes], dtype=int)
    return bank, max(len(p) for p in tapes)


class CarEnv:
    def __init__(
        self,
        num_worlds=4096,
        seed=42,
        render=False,
        tracks: list[Track] | None = None,
        reward: RewardConfig | None = None,
        episode_steps: int | None = None,
        evaluate: bool = False,
    ):
        """evaluate: world i always drives tracks[i % len(tracks)] from the same
        clean start (see EnvParams.evaluate). For eval.py and view.py."""
        # Torch and Warp share one stream so they never race each other. It's a
        # torch-made (non-blocking) stream: PyTorch can't record CUDA graphs on
        # Warp's default stream, which is a blocking one
        wp.init()
        stream = torch.cuda.Stream()
        wp.set_stream(wp.stream_from_torch(stream))
        torch.cuda.set_stream(stream)

        self.num_worlds = num_worlds
        self.obs_dim = OBS_DIM
        self.action_dim = 2
        self.seed = seed
        self.render = render
        # False when a caller records env.step() into its own CUDA graph (see
        # ppo/graph.py): a graph can't be launched while another is being recorded
        self.use_graph = True

        cfg = load_config()
        # Track arrays go to the GPU once; kernels only read them
        self.tracks = tracks or [Track.from_centerline(oval_centerline())]
        self.bank, self.num_tape = make_track_bank(self.tracks)
        # One shared set of tape boxes, moved onto each world's track by write_tape
        _, self.model = build_model(cfg, tape_pool=self.num_tape)
        self.first_tape = self.model.geom("tape_0").id
        self.device_model = mjw.put_model(self.model)
        self.device_data = mjw.make_data(self.model, nworld=num_worlds)

        self.car = CarParams()
        self.car.wheelbase = cfg.geometry.wheelbase
        self.car.max_steer = cfg.steering.max_angle
        self.car.max_speed = cfg.throttle.max_speed
        self.car.dt = cfg.control.dt

        # Anything not given comes from experiments/baseline.toml
        if reward is None or episode_steps is None:
            baseline = load_experiment()
            reward = reward or baseline.reward
            episode_steps = episode_steps or baseline.env.episode_steps
        self.params = EnvParams()
        self.params.episode_steps = episode_steps
        self.params.evaluate = int(evaluate)
        self.params.w_progress = reward.progress
        self.params.w_off_track = reward.off_track
        self.params.w_center = reward.center
        self.params.w_steer_rate = reward.steer_rate

        def zeros(shape, dtype=float):
            arr = wp.zeros(shape, dtype=dtype)
            return arr, wp.to_torch(arr)

        self.actions, self.actions_t = zeros((num_worlds, self.action_dim))
        self.obs, self.obs_t = zeros((num_worlds, OBS_DIM))
        self.final_obs, self.final_obs_t = zeros((num_worlds, OBS_DIM))
        self.reward, self.reward_t = zeros(num_worlds)
        # Each reward term separately, for logging (order in env/kernels.py)
        self.reward_terms, self.reward_terms_t = zeros((num_worlds, NUM_REWARD_TERMS))
        self.laps, self.laps_t = zeros(num_worlds)  # laps driven this step
        self.done, self.done_t = zeros(num_worlds)  # 1.0 == done
        self.truncated, self.truncated_t = zeros(num_worlds)

        # Per-world car and track state
        self.idx, _ = zeros(num_worlds, int)  # nearest centerline point
        self.s_prev, _ = zeros(num_worlds)  # distance along the track last step
        self.speed, _ = zeros(num_worlds)
        self.steer, _ = zeros(num_worlds)
        self.steps, _ = zeros(num_worlds, int)
        self.resets, _ = zeros(num_worlds, int)
        self.track_id, self.track_id_t = zeros(num_worlds, int)  # index into tracks

        self.rgb_out_t = None
        self.pixels_t = None
        self.pixel_shape = None
        if render:
            cam = self.model.camera("front")
            self.cam_w, self.cam_h = self.model.cam_resolution[cam.id]
            # Only render the policy's camera, not the chase camera
            self.rc = mjw.create_render_context(
                self.model,
                num_worlds,
                render_rgb=True,
                render_depth=False,
                cam_active=["front"],
            )
            self.rgb_out = wp.zeros((num_worlds, self.cam_h, self.cam_w), dtype=wp.vec3)
            self.rgb_hwc_t = wp.to_torch(self.rgb_out)  # (N, H, W, 3), as rendered
            self.rgb_out_t = self.rgb_hwc_t.permute(0, 3, 1, 2)  # (N, 3, H, W) view
            self.cam_index = cam.id
            self.pixel_shape = (
                3,
                int(self.cam_h) - crop_top(int(self.cam_h)),
                int(self.cam_w),
            )
            self.pixels_t = torch.zeros(
                (num_worlds, *self.pixel_shape), dtype=torch.uint8, device="cuda"
            )

        # Record one step into a CUDA graph, then replay it every step
        self._step_ops()
        with wp.ScopedCapture() as capture:
            self._step_ops()
        self.graph = capture.graph

    def reset(self) -> ResetOutput:
        wp.launch(
            reset_all,
            dim=self.num_worlds,
            inputs=[
                self.device_data.qpos,
                self.bank,
                self.track_id,
                self.car,
                self.params,
                self.idx,
                self.s_prev,
                self.speed,
                self.steer,
                self.obs,
                self.reward,
                self.done,
                self.truncated,
                self.steps,
                self.resets,
                self.seed,
                self.num_worlds,
            ],
        )
        if self.render:
            self._render()
            self._preprocess()
        return ResetOutput(obs=self.obs_t, rgb=self.rgb_out_t, pixels=self.pixels_t)

    def step(self, actions: torch.Tensor) -> StepOutput:
        self.actions_t.copy_(actions)
        if self.use_graph:
            wp.capture_launch(self.graph)  # pyright: ignore[reportArgumentType]
        else:
            self._step_ops()
        if self.render:
            self._preprocess()
        return StepOutput(
            obs=self.obs_t,
            rgb=self.rgb_out_t,
            pixels=self.pixels_t,
            reward=self.reward_t,
            done=self.done_t,
            truncated=self.truncated_t,
            final_obs=self.final_obs_t,
        )

    def _step_ops(self):
        wp.launch(
            step,
            dim=self.num_worlds,
            inputs=[
                self.actions,
                self.device_data.qpos,
                self.bank,
                self.track_id,
                self.car,
                self.params,
                self.idx,
                self.s_prev,
                self.speed,
                self.steer,
                self.obs,
                self.reward,
                self.reward_terms,
                self.laps,
                self.done,
                self.truncated,
                self.final_obs,
                self.steps,
                self.resets,
                self.seed,
                self.num_worlds,
            ],
        )
        if self.render:
            self._render()

    def _preprocess(self):
        assert self.pixels_t is not None
        preprocess(self.rgb_hwc_t, self.pixels_t)

    def _render(self):
        # No physics step: kinematics turns qpos into body and geom poses, then
        # com_pos + camlight move the cameras and lights with them (without these
        # the camera stays at its initial pose and renders black)
        mjw.kinematics(self.device_model, self.device_data)
        wp.launch(
            write_tape,
            dim=(self.num_worlds, self.num_tape),
            inputs=[
                self.bank,
                self.track_id,
                self.first_tape,
                self.device_data.geom_xpos,
                self.device_data.geom_xmat,
            ],
        )
        mjw.com_pos(self.device_model, self.device_data)
        mjw.camlight(self.device_model, self.device_data)
        mjw.refit_bvh(self.device_model, self.device_data, self.rc)
        mjw.render(self.device_model, self.device_data, self.rc)
        mjw.get_rgb(self.rc, self.cam_index, self.rgb_out)  # pyright: ignore[reportArgumentType]
