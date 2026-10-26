import torch

from env.kernels import CMD_STEER


class RolloutBuffer:
    """T steps x W worlds of experience, all on the GPU. Shapes are (T, W, ...).

    pixels (the actor's camera frames) are stored as uint8: 4x less memory than
    float, which is what limits how many worlds fit on the GPU.
    """

    def __init__(
        self, rollout_steps, num_worlds, obs_dim, action_dim, pixel_shape
    ) -> None:
        T, W = rollout_steps, num_worlds
        self.rollout_steps = T
        self.num_worlds = W

        def zeros(*shape):
            return torch.zeros(shape, device="cuda")

        self.obs = zeros(T, W, obs_dim)  # sim state, for the critic
        self.pixels = torch.zeros(
            (T, W, *pixel_shape), dtype=torch.uint8, device="cuda"
        )
        self.actions = zeros(T, W, action_dim)
        self.log_probs = zeros(T, W)
        self.values = zeros(T, W)
        self.rewards = zeros(T, W)
        self.dones = zeros(T, W)
        self.advantages = zeros(T, W)
        self.returns = zeros(T, W)

    def collect_rollout(self, env, actor, critic, reward_scale, gamma):
        """Run the policy for T steps in every world. Returns stats.

        Reads the env's fixed output tensors (env.pixels_t, env.obs_t), which each
        step overwrites in place, so this can be recorded in a CUDA graph.

        Stats are rates and per-step averages, not per-episode numbers: a rollout
        (1.6 s) often contains no finished episodes, especially when many worlds
        time out together. They stay on the GPU as tensors; reading them (.item())
        forces a sync, so the caller only does that when it logs.
        """
        # [crashes, laps, speed, |offset|, |steer change|, then one per reward term]
        stats = torch.zeros(5 + env.reward_terms_t.shape[1], device="cuda")

        with torch.no_grad():
            obs, pixels = env.obs_t, env.pixels_t
            for step in range(self.rollout_steps):
                dist = actor.get_dist(pixels)  # the actor only sees the camera
                action = dist.sample()
                self.obs[step] = obs
                self.pixels[step] = pixels
                self.actions[step] = action
                self.log_probs[step] = dist.log_prob(action).sum(-1)
                self.values[step] = critic(obs)

                out = env.step(action.clamp(-1.0, 1.0))
                reward, done, truncated = out.reward, out.done, out.truncated

                # Timeout isn't failure: add the value of where the car would have
                # gone next, so cutting the episode short doesn't look like a crash
                bootstrap = critic(out.final_obs) * truncated
                self.rewards[step] = reward * reward_scale + gamma * bootstrap
                self.dones[step] = done

                stats += torch.cat(
                    [
                        torch.stack(
                            [
                                (done * (1 - truncated)).sum(),  # left the track
                                env.laps_t.sum(),  # laps (each world's own track)
                                obs[:, 3].sum(),  # speed / max_speed
                                obs[:, 0].abs().sum(),  # |offset|, 1 = on the tape
                                # |steering command change| this step, in [-1, 1]
                                # units (obs[:, CMD_STEER], env/kernels.py).
                                # `obs` is the env's output tensor, already
                                # overwritten by this step, so "before" comes from
                                # the buffer copy. Skipped on resets (steering -> 0)
                                (
                                    (
                                        out.obs[:, CMD_STEER]
                                        - self.obs[step, :, CMD_STEER]
                                    ).abs()
                                    * (1 - done)
                                ).sum(),
                            ]
                        ),
                        env.reward_terms_t.sum(0),
                    ]
                )

        samples = self.rollout_steps * self.num_worlds
        minutes = samples * env.car.dt / 60  # total driving time across all worlds
        terms = stats[5:] / samples  # average per step
        return {
            "drive/crashes_per_min": stats[0] / minutes,
            "drive/laps_per_min": stats[1] / minutes,
            "drive/speed": stats[2] / samples * env.car.max_speed,
            "drive/offset_abs": stats[3] / samples,
            "drive/steer_change": stats[4] / samples,
            "reward/total": terms.sum(),
            "reward/progress": terms[0],
            "reward/off_track": terms[1],
            "reward/center": terms[2],
            "reward/steer_rate": terms[3],
        }

    def compute_gae(self, next_value, gamma, gae_lambda):
        """Advantages: how much better each action did than the critic expected.

        Walks backwards in time. (1 - done) stops future rewards from leaking
        across an episode boundary into the previous episode.
        """
        next_adv = torch.zeros(self.num_worlds, device="cuda")
        for step in reversed(range(self.rollout_steps)):
            next_val = (
                next_value if step == self.rollout_steps - 1 else self.values[step + 1]
            )
            not_done = 1 - self.dones[step]
            delta = self.rewards[step] + gamma * next_val * not_done - self.values[step]
            next_adv = delta + gamma * gae_lambda * not_done * next_adv
            self.advantages[step] = next_adv
        torch.add(self.advantages, self.values, out=self.returns)
