import torch

from rcsim.env.kernels import CMD_STEER


def ppo_update(
    actor,
    critic,
    optimizer,
    buffer,
    clip_epsilon=0.2,
    critic_loss_weight=0.5,
    entropy_weight=0.01,
    update_epochs=4,
    minibatch_size=32768,
    max_grad_norm=0.5,
):
    # Keep (T, W, ...) unflattened so each world's sequence stays in order for the GRU
    obs = buffer.obs  # critic input (sim state)
    pixels = buffer.pixels  # actor input (camera)
    actions = buffer.actions
    old_log_probs = buffer.log_probs
    returns = buffer.returns
    advantages = buffer.advantages
    h0 = buffer.h0
    dones = buffer.dones
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    T, W = buffer.rollout_steps, buffer.num_worlds

    params = list(actor.parameters()) + list(critic.parameters())
    worlds_per_mb = minibatch_size // T
    stats = torch.zeros(3, device="cuda")
    num_updates = 0

    for _ in range(update_epochs):
        # Shuffle whole worlds (keeping each world's T steps chronological);
        # argsort of noise is safe to record in a CUDA graph (rcsim/ppo/graph.py)
        indices = torch.rand(W, device="cuda").argsort()
        for start in range(0, W, worlds_per_mb):
            mb = indices[start : start + worlds_per_mb]

            prev_cmd = obs[:, mb, CMD_STEER : CMD_STEER + 2]
            dist = actor.sequence(pixels[:, mb], prev_cmd, dones[:, mb], h0[mb])
            new_log_probs = dist.log_prob(actions[:, mb]).sum(-1)
            entropy = dist.entropy().sum(-1).mean()

            # How much more (or less) likely the action is now than when collected
            ratio = torch.exp(new_log_probs - old_log_probs[:, mb])
            clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
            actor_loss = -torch.min(
                ratio * advantages[:, mb], clipped * advantages[:, mb]
            ).mean()

            critic_loss = ((critic(obs[:, mb]) - returns[:, mb]) ** 2).mean()

            loss = (
                actor_loss + critic_loss_weight * critic_loss - entropy_weight * entropy
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()

            stats += torch.stack(
                [actor_loss.detach(), critic_loss.detach(), entropy.detach()]
            )
            num_updates += 1

    stats /= num_updates
    return {"actor_loss": stats[0], "critic_loss": stats[1], "entropy": stats[2]}
