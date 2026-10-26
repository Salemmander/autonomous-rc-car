import torch


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
    # (T, W, ...) -> (T*W, ...): every step of every world is one sample
    obs = buffer.obs.flatten(0, 1)  # critic input (sim state)
    pixels = buffer.pixels.flatten(0, 1)  # actor input (camera)
    actions = buffer.actions.flatten(0, 1)
    old_log_probs = buffer.log_probs.flatten()
    returns = buffer.returns.flatten()
    advantages = buffer.advantages.flatten()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    params = list(actor.parameters()) + list(critic.parameters())
    batch_size = obs.shape[0]
    stats = torch.zeros(3, device="cuda")
    num_updates = 0

    for _ in range(update_epochs):
        # Random order via argsort of noise: same as randperm, but safe to
        # record in a CUDA graph (see ppo/graph.py)
        indices = torch.rand(batch_size, device="cuda").argsort()
        for start in range(0, batch_size, minibatch_size):
            mb = indices[start : start + minibatch_size]

            dist = actor.get_dist(pixels[mb])
            new_log_probs = dist.log_prob(actions[mb]).sum(-1)
            entropy = dist.entropy().sum(-1).mean()

            # How much more (or less) likely the action is now than when collected
            ratio = torch.exp(new_log_probs - old_log_probs[mb])
            clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
            actor_loss = -torch.min(
                ratio * advantages[mb], clipped * advantages[mb]
            ).mean()

            critic_loss = ((critic(obs[mb]) - returns[mb]) ** 2).mean()

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
