"""GPU-parallel car env, the car version of parallel-cartpole's CartPole.

Every world drives the same track for now. Actions are [steer, throttle] in
[-1, 1]; observations are listed in env/kernels.py:write_obs.
"""

from dataclasses import dataclass

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from assets.car import build_model
from assets.track import Track, oval_centerline
from config import load_config
from env.kernels import OBS_DIM, CarParams, TrackData, reset_all, step


@dataclass
class StepOutput:
    obs: torch.Tensor
    rgb: torch.Tensor | None
    reward: torch.Tensor
    done: torch.Tensor
    truncated: torch.Tensor
    final_obs: torch.Tensor


@dataclass
class ResetOutput:
    obs: torch.Tensor
    rgb: torch.Tensor | None


def _vec2(a: np.ndarray) -> wp.array:
    return wp.array(a.astype(np.float32), dtype=wp.vec2)


class CarEnv:
    def __init__(
        self, num_worlds=4096, seed=42, render=False, track: Track | None = None
    ):
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
        self.track = track or Track.from_centerline(oval_centerline())
        _, self.model = build_model(cfg, self.track)
        self.device_model = mjw.put_model(self.model)
        self.device_data = mjw.make_data(self.model, nworld=num_worlds)

        # Track arrays go to the GPU once; kernels only read them
        self.track_data = TrackData()
        self.track_data.centerline = _vec2(self.track.centerline)
        self.track_data.tangents = _vec2(self.track.tangents)
        self.track_data.normals = _vec2(self.track.normals)
        self.track_data.s = wp.array(self.track.s.astype(np.float32), dtype=float)
        self.track_data.n = len(self.track.centerline)
        self.track_data.length = self.track.length
        self.track_data.half_width = self.track.width / 2

        self.car = CarParams()
        self.car.wheelbase = cfg.geometry.wheelbase
        self.car.max_steer = cfg.steering.max_angle
        self.car.max_speed = cfg.throttle.max_speed
        self.car.dt = cfg.control.dt

        def zeros(shape, dtype=float):
            arr = wp.zeros(shape, dtype=dtype)
            return arr, wp.to_torch(arr)

        self.actions, self.actions_t = zeros((num_worlds, self.action_dim))
        self.obs, self.obs_t = zeros((num_worlds, OBS_DIM))
        self.final_obs, self.final_obs_t = zeros((num_worlds, OBS_DIM))
        self.reward, self.reward_t = zeros(num_worlds)
        self.done, self.done_t = zeros(num_worlds)  # 1.0 == done
        self.truncated, self.truncated_t = zeros(num_worlds)

        # Per-world car and track state
        self.idx, _ = zeros(num_worlds, int)  # nearest centerline point
        self.s_prev, _ = zeros(num_worlds)  # distance along the track last step
        self.speed, _ = zeros(num_worlds)
        self.steer, _ = zeros(num_worlds)
        self.steps, _ = zeros(num_worlds, int)
        self.resets, _ = zeros(num_worlds, int)

        self.rgb_out_t = None
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
            self.rgb_out_t = wp.to_torch(self.rgb_out).permute(0, 3, 1, 2)
            self.cam_index = cam.id

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
                self.track_data,
                self.car,
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
        return ResetOutput(obs=self.obs_t, rgb=self.rgb_out_t)

    def step(self, actions: torch.Tensor) -> StepOutput:
        self.actions_t.copy_(actions)
        if self.use_graph:
            wp.capture_launch(self.graph)  # pyright: ignore[reportArgumentType]
        else:
            self._step_ops()
        return StepOutput(
            obs=self.obs_t,
            rgb=self.rgb_out_t,
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
                self.track_data,
                self.car,
                self.idx,
                self.s_prev,
                self.speed,
                self.steer,
                self.obs,
                self.reward,
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

    def _render(self):
        # No physics step: kinematics turns qpos into geom and camera poses
        mjw.kinematics(self.device_model, self.device_data)
        mjw.refit_bvh(self.device_model, self.device_data, self.rc)
        mjw.render(self.device_model, self.device_data, self.rc)
        mjw.get_rgb(self.rc, self.cam_index, self.rgb_out)  # pyright: ignore[reportArgumentType]
