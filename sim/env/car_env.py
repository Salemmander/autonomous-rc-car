"""GPU-parallel car env, the car version of parallel-cartpole's CartPole.

Each world drives its own track, picked at random from `tracks` on every reset.
Actions are [steer, throttle] in [-1, 1].

Two kinds of observation:
- pixels (render=True): the car's camera, preprocessed exactly like on the Pi
  (env/preprocess.py). This is all the actor sees. In training, the look is
  randomized every episode ([visuals]: floor, tape, camera, lighting, exposure, noise).
- obs: sim-only state (env/kernels.py:write_obs). Only the critic sees it, during
  training; the real car never has it.
"""

from dataclasses import dataclass

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from assets.car import LAMPS, build_model
from assets.track import Track, oval_centerline, tape_poses
from config import load_config
from env.kernels import (
    HISTORY,
    NUM_REWARD_TERMS,
    OBS_DIM,
    CarParams,
    EnvParams,
    LookIds,
    TrackBank,
    Worlds,
    postprocess,
    randomize_look,
    reset_all,
    step,
    write_tape,
)
from env.preprocess import crop_top
from experiment import RandomizeConfig, RewardConfig, VisualsConfig, load_experiment


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


# Parts of the model that randomize_look and write_tape change per world
PER_WORLD_MODEL_FIELDS = (
    "mat_texid",  # floor texture
    "mat_rgba",  # floor and tape color
    "mat_texrepeat",  # floor pattern scale
    "geom_size",  # tape width
    "cam_pos",  # camera height
    "cam_quat",  # camera tilt
    "mat_specular",  # floor shine
    "mat_shininess",
    "light_pos",  # sun and lamps
    "light_dir",
    "light_diffuse",
    "light_ambient",
    "light_specular",
)


class CarEnv:
    def __init__(
        self,
        num_worlds=4096,
        seed=42,
        render=False,
        tracks: list[Track] | None = None,
        reward: RewardConfig | None = None,
        episode_steps: int | None = None,
        randomize: RandomizeConfig | None = None,
        visuals: VisualsConfig | None = None,
        evaluate: bool = False,
        nominal: bool | None = None,
    ):
        """evaluate: world i always drives tracks[i % len(tracks)] from the same
        clean start (see EnvParams.evaluate). For eval.py and view.py.
        nominal: the car.toml car and the default look exactly, instead of
        randomizing them every episode. Default: on when evaluating, off when
        training."""
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
        # Model arrays that differ per world (the look, randomize_look). The
        # renderer reads row (world % rows), so one row means "shared by all"
        for name in PER_WORLD_MODEL_FIELDS:
            shared = getattr(self.device_model, name)
            per_world = np.repeat(shared.numpy()[:1], num_worlds, axis=0)
            setattr(self.device_model, name, wp.array(per_world, dtype=shared.dtype))

        self.car = CarParams()
        self.car.wheelbase = cfg.geometry.wheelbase
        self.car.max_steer = cfg.steering.max_angle
        self.car.max_speed = cfg.throttle.max_speed
        self.car.dt = cfg.control.dt
        self.car.latency = cfg.control.latency_s
        self.car.steer_time = cfg.steering.response_time_s
        self.car.speed_time = cfg.throttle.response_time_s

        # Anything not given comes from experiments/baseline.toml
        if None in (reward, episode_steps, randomize, visuals):
            baseline = load_experiment()
            reward = reward or baseline.reward
            episode_steps = episode_steps or baseline.env.episode_steps
            randomize = randomize or baseline.randomize
            visuals = visuals or baseline.visuals
        assert reward and episode_steps and randomize and visuals
        self.params = EnvParams()
        self.params.episode_steps = episode_steps
        self.params.evaluate = int(evaluate)
        self.params.nominal = int(evaluate if nominal is None else nominal)
        self.params.w_progress = reward.progress
        self.params.w_off_track = reward.off_track
        self.params.w_center = reward.center
        self.params.w_steer_rate = reward.steer_rate
        self.params.r_latency = wp.vec2(*randomize.latency)
        self.params.r_steer_time = wp.vec2(*randomize.steer_response)
        self.params.r_speed_time = wp.vec2(*randomize.speed_response)
        self.params.r_max_speed = wp.vec2(*randomize.max_speed)
        self.params.r_max_steer = wp.vec2(*randomize.max_steer)
        self.params.r_steer_trim = randomize.steer_trim
        self.params.v_floor_brightness = wp.vec2(*visuals.floor_brightness)
        self.params.v_floor_saturation = visuals.floor_saturation
        self.params.v_floor_scale = wp.vec2(*visuals.floor_scale)
        self.params.v_tape_width = wp.vec2(*visuals.tape_width)
        self.params.v_tape_white = visuals.tape_white
        self.params.v_camera_height = visuals.camera_height
        self.params.v_camera_pitch = np.radians(visuals.camera_pitch_deg)
        self.params.v_exposure = wp.vec2(*visuals.exposure)
        self.params.v_contrast = wp.vec2(*visuals.contrast)
        self.params.v_white_balance = visuals.white_balance
        self.params.v_noise = visuals.noise
        self.params.v_sun = wp.vec2(*visuals.sun)
        self.params.v_sun_elevation = wp.vec2(*np.radians(visuals.sun_elevation_deg))
        self.params.v_ambient = wp.vec2(*visuals.ambient)
        self.params.v_lamp = wp.vec2(*visuals.lamp)
        self.params.v_lamp_height = wp.vec2(*visuals.lamp_height)
        self.params.v_light_tint = visuals.light_tint
        self.params.v_floor_gloss = wp.vec2(*visuals.floor_gloss)

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

        # Per-world car and track state (see Worlds in env/kernels.py)
        self.worlds = Worlds()
        for name in ("idx", "steps", "resets", "track_id"):
            setattr(self.worlds, name, wp.zeros(num_worlds, dtype=int))
        for name in (
            "s_prev", "speed", "steer", "latency", "steer_time", "speed_time",
            "max_speed", "max_steer", "steer_trim", "tape_width", "exposure",
            "contrast", "noise",
        ):  # fmt: skip
            setattr(self.worlds, name, wp.zeros(num_worlds, dtype=float))
        self.worlds.commands = wp.zeros((num_worlds, HISTORY), dtype=wp.vec2)
        self.worlds.color_gain = wp.zeros(num_worlds, dtype=wp.vec3)

        # Where the randomized look goes in the model
        m = self.model
        floor_mat = m.material("floor").id
        tape_mat = m.material("tape").id
        cam_id = m.camera("front").id
        self.look_ids = LookIds()
        self.look_ids.floor_mat = floor_mat
        self.look_ids.tape_mat = tape_mat
        self.look_ids.num_floor_tex = m.ntex  # every texture is a floor; 0 = default
        self.look_ids.texrepeat = wp.vec2(*m.mat_texrepeat[floor_mat])
        self.look_ids.tape_rgba = wp.vec4(*m.mat_rgba[tape_mat])
        self.look_ids.cam = cam_id
        self.look_ids.cam_pos = wp.vec3(*m.cam_pos[cam_id])
        self.look_ids.cam_quat = wp.quat(*m.cam_quat[cam_id])
        sun = m.light("sun").id
        self.look_ids.sun = sun
        self.look_ids.sun_dir = wp.vec3(*m.light_dir[sun])
        self.look_ids.sun_diffuse = wp.vec3(*m.light_diffuse[sun])
        self.look_ids.sun_ambient = wp.vec3(*m.light_ambient[sun])
        self.look_ids.sun_specular = wp.vec3(*m.light_specular[sun])
        self.look_ids.first_lamp = m.light("lamp_0").id
        self.look_ids.num_lamps = LAMPS
        self.look_ids.floor_specular = float(m.mat_specular[floor_mat])
        self.look_ids.floor_shininess = float(m.mat_shininess[floor_mat])
        self.track_id_t = wp.to_torch(self.worlds.track_id)

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
            self.pixels = wp.zeros((num_worlds, *self.pixel_shape), dtype=wp.uint8)
            self.pixels_t = wp.to_torch(self.pixels)

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
                self.car,
                self.params,
                self.worlds,
                self.obs,
                self.reward,
                self.done,
                self.truncated,
                self.seed,
                self.num_worlds,
            ],
        )
        if self.render:
            self._randomize_look(only_done=False)
            self._render()
        return ResetOutput(obs=self.obs_t, rgb=self.rgb_out_t, pixels=self.pixels_t)

    def step(self, actions: torch.Tensor) -> StepOutput:
        self.actions_t.copy_(actions)
        if self.use_graph:
            wp.capture_launch(self.graph)  # pyright: ignore[reportArgumentType]
        else:
            self._step_ops()
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
                self.car,
                self.params,
                self.worlds,
                self.obs,
                self.reward,
                self.reward_terms,
                self.laps,
                self.done,
                self.truncated,
                self.final_obs,
                self.seed,
                self.num_worlds,
            ],
        )
        if self.render:
            self._randomize_look(only_done=True)  # worlds that just reset
            self._render()

    def _randomize_look(self, only_done: bool):
        m = self.device_model
        wp.launch(
            randomize_look,
            dim=self.num_worlds,
            inputs=[
                self.params,
                self.worlds,
                self.look_ids,
                self.done,
                int(only_done),
                self.seed,
                self.num_worlds,
                m.mat_texid,
                m.mat_rgba,
                m.mat_texrepeat,
                m.cam_pos,
                m.cam_quat,
                m.mat_specular,
                m.mat_shininess,
                m.light_pos,
                m.light_dir,
                m.light_diffuse,
                m.light_ambient,
                m.light_specular,
            ],
        )

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
                self.worlds,
                self.first_tape,
                self.device_data.geom_xpos,
                self.device_data.geom_xmat,
                self.device_model.geom_size,
            ],
        )
        mjw.com_pos(self.device_model, self.device_data)
        mjw.camlight(self.device_model, self.device_data)
        mjw.refit_bvh(self.device_model, self.device_data, self.rc)
        mjw.render(self.device_model, self.device_data, self.rc)
        mjw.get_rgb(self.rc, self.cam_index, self.rgb_out)  # pyright: ignore[reportArgumentType]
        assert self.pixel_shape is not None
        wp.launch(
            postprocess,
            dim=(self.num_worlds, self.pixel_shape[1], self.pixel_shape[2]),
            inputs=[
                self.rgb_out,
                self.worlds,
                crop_top(int(self.cam_h)),
                self.seed,
                self.pixels,
            ],
        )
