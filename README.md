# CartPole — REINFORCE Policy Gradient

> **University Learning Project** — bridging the gap between theoretical knowledge and practical implementation of Reinforcement Learning.

This repository implements the **REINFORCE** policy-gradient algorithm (Williams, 1992) applied to the classic **CartPole-v1** environment from [Gymnasium](https://gymnasium.farama.org/).

The goal is not just to solve CartPole, but to make every line of code understandable. Comments explain *why* things are done, not just *what* they do.

---

## What is REINFORCE?

REINFORCE is the simplest policy-gradient algorithm.  
The core idea: *make actions that led to high returns more likely; make actions that led to low returns less likely.*

Formally, we maximise the expected return J(θ) by following the gradient:

```
∇J(θ) = E[ ∇log π(a|s, θ) · G_t ]
```

where **G_t** is the discounted cumulative reward from step *t* onward.

---

## Repository Structure

```
.
├── main.py          # Entry point — runs training and records the video
├── model.py         # Policy network (MLP with Softmax output)
├── train.py         # REINFORCE training loop
├── pyproject.toml   # Project metadata and dependencies (managed by uv)
└── README.md
```

---

## Getting Started

### 1. Install `uv` (if you haven't already)

```bash
pip install uv
```

### 2. Create the virtual environment and install dependencies

```bash
uv sync
```

### 3. Run training

```bash
uv run python main.py
```

This will:
- Train the policy for 1 000 episodes
- Save a learning-curve plot (`learning_curve.png`)
- Save the trained weights (`policy_weights.pth`)
- Record an evaluation video in the `videos/` folder

---

## Environment: CartPole-v1

A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The goal is to keep the pole balanced by pushing the cart left or right.

| | |
|---|---|
| **Observation** | Cart pos, cart vel, pole angle, pole angular vel |
| **Actions** | Push left (0) or push right (1) |
| **Reward** | +1 for every timestep the pole stays up |
| **Episode end** | Pole angle > ±12° · Cart pos > ±2.4 · Steps > 500 |

---

## Key Concepts Implemented

- **Policy network**: a two-layer MLP with Softmax output representing a probability distribution over actions.
- **Categorical distribution**: used to sample actions stochastically during training, enabling exploration.
- **Discounted returns**: `G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + …`
- **Return normalisation**: subtracting the mean and dividing by the standard deviation reduces training variance.
- **Policy gradient loss**: `L = -Σ log π(a_t|s_t) · G_t`
- **Device-agnostic code**: automatically runs on GPU if available, otherwise falls back to CPU.

---

## References

- Williams, R. J. (1992). *Simple statistical gradient-following algorithms for connectionist reinforcement learning.* Machine Learning, 8(3–4), 229–256.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- [Gymnasium documentation](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
