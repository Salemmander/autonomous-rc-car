import torch


class RolloutBuffer:
    """T steps x W worlds of experience, all on the GPU. Shapes are (T, W, ...)."""

    def __init__(self, rollout_steps, num_worlds, obs_dim, action_dim) -> None:
        T, W = rollout_steps, num_worlds
        self.rollout_steps = T
        self.num_worlds = W

        def zeros(*shape):
            return torch.zeros(shape, device="cuda")

        self.obs = zeros(T, W, obs_dim)
        self.actions = zeros(T, W, action_dim)
        self.log_probs = zeros(T, W)
        self.values = zeros(T, W)
        self.rewards = zeros(T, W)
        self.dones = zeros(T, W)
        self.advantages = zeros(T, W)
        self.returns = zeros(T, W)

        # Per-world running totals for the episode in progress (for logging)
        self.ep_return = zeros(W)
        self.ep_len = zeros(W)
        self.ep_progress = zeros(W)  # meters along the track

    def collect_rollout(self, env, actor, critic, obs, reward_scale, gamma):
        """Run the policy for T steps in every world. Returns the last obs and stats.

        Stats stay on the GPU as tensors; reading them (.item()) forces a sync, so
        the caller only does that when it prints.
        """
        meters_per_reward = env.car.max_speed * env.car.dt
        stats = torch.zeros(6, device="cuda")  # see the names in the return below

        with torch.no_grad():
            for step in range(self.rollout_steps):
                dist = actor.get_dist(obs)
                action = dist.sample()
                self.obs[step] = obs
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

                off_track = done * (1 - truncated)
                self.ep_return += reward
                self.ep_len += 1
                self.ep_progress += (
                    reward.clamp(min=0) * meters_per_reward * (1 - off_track)
                )
                stats += torch.stack(
                    [
                        done.sum(),
                        off_track.sum(),
                        (self.ep_return * done).sum(),
                        (self.ep_len * done).sum(),
                        (self.ep_progress * done).sum(),
                        obs[:, 3].sum(),  # speed / max_speed
                    ]
                )
                keep = 1 - done
                self.ep_return *= keep
                self.ep_len *= keep
                self.ep_progress *= keep

                obs = out.obs

        finished = stats[0].clamp(min=1)
        return obs, {
            "episodes": stats[0],
            "off_track_frac": stats[1] / finished,
            "return": stats[2] / finished,
            "length": stats[3] / finished,
            "laps": stats[4] / finished / env.track.length,
            "speed": stats[5]
            / (self.rollout_steps * self.num_worlds)
            * env.car.max_speed,
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
