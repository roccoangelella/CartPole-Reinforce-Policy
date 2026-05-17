# CartPole — REINFORCE Policy Gradient

This repo is a practical implementation of the REINFORCE policy gradient algorithm applied to the CartPole-v1 environment.

It comes from the **HuggingFace Deep Reinforcement Learning Course**, specifically the unit on REINFORCE policy gradient. The goal is to bridge the gap between the theoretical knowledge of policy gradient methods and actually putting them into practice as a university learning project.

---

## Structure

```
.
├── main.py        # runs training and records the evaluation video
├── train.py       # the REINFORCE training loop
├── model.py       # the MLP policy network
├── pyproject.toml # dependencies managed with uv
└── README.md
```

---

## Getting Started

Install [uv](https://github.com/astral-sh/uv) if you haven't already:

```bash
pip install uv
```

Create the environment and install dependencies:

```bash
uv sync
```

Run training:

```bash
uv run python main.py
```

---

## References

- [HuggingFace Deep RL Course — Unit on REINFORCE](https://huggingface.co/learn/deep-rl-course)
- Williams, R. J. (1992). *Simple statistical gradient-following algorithms for connectionist reinforcement learning.* Machine Learning.
