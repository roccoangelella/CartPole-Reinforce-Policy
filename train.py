import numpy as np
import torch
from torch import optim


def compute_returns(rewards, gamma):
    returns = []
    G = 0

    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)
    return returns


def reinforce(
    policy,
    env,
    device,
    n_episodes=1000,
    gamma=1.0,
    lr=1e-2,
    print_every=100,
):
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    scores = []

    for episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        rewards = []
        log_probs = []

        done = False
        while not done:
            action, log_prob = policy.act(state, device)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)

        returns = compute_returns(rewards, gamma)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)

        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        scores.append(total_reward)

        if episode % print_every == 0:
            avg = np.mean(scores[-print_every:])
            print(
                f"Episode {episode}/{n_episodes} | "
                f"Avg reward (last {print_every}): {avg:.1f}"
            )

    return scores
