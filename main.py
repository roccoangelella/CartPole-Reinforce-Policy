"""A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left and right direction on the cart.

The episode ends if:

The pole Angle is greater than ±12°
The Cart Position is greater than ±2.4
The episode length is greater than 500

We get a reward of +1 every timestep that the Pole stays in the equilibrium.
"""

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.model import Policy
from src.train import reinforce
from src.config import (
    ENV_NAME,
    HIDDEN_SIZE,
    N_EPISODES,
    GAMMA,
    LR,
    PRINT_EVERY,
    device,
)

np.bool8 = np.bool_

print(f"Using device: {device}")


def plot_scores(scores):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(scores, alpha=0.4, label="Episode score")

    window = 100
    if len(scores) >= window:
        rolling_mean = np.convolve(scores, np.ones(window) / window, mode="valid")
        ax.plot(
            range(window - 1, len(scores)),
            rolling_mean,
            linewidth=2,
            label=f"Rolling mean ({window} ep)",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title("REINFORCE on CartPole-v1 — Learning Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=150)
    plt.show()


def record_video(policy, n_steps=500):
    video_dir = "videos"

    eval_env = gym.make(ENV_NAME, render_mode="rgb_array")
    eval_env = gym.wrappers.RecordVideo(
        eval_env,
        video_folder=video_dir,
        episode_trigger=lambda ep: True,
    )

    state, _ = eval_env.reset()
    total_reward = 0

    for _ in range(n_steps):
        action, _ = policy.act(state, device)
        state, reward, terminated, truncated, _ = eval_env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    eval_env.close()
    print(f"Evaluation episode — total reward: {total_reward:.1f}")
    print(f"Video saved to '{video_dir}/' folder.")


def main():
    env = gym.make(ENV_NAME)

    obs_space = env.observation_space
    act_space = env.action_space

    print(obs_space, act_space)

    state_size  = obs_space.shape[0]
    action_size = act_space.n

    policy = Policy(state_size, action_size, HIDDEN_SIZE).to(device)

    scores = reinforce(
        policy=policy,
        env=env,
        device=device,
        n_episodes=N_EPISODES,
        gamma=GAMMA,
        lr=LR,
        print_every=PRINT_EVERY,
    )
    env.close()

    plot_scores(scores)

    torch.save(policy.state_dict(), "policy_weights.pth")

    record_video(policy)


if __name__ == "__main__":
    main()
